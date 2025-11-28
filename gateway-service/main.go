package main

import (
	"context"
	"io"
	"log"
	"net/http"
	"os"
	"time"

	feedv1 "github.com/arietids/video-rss-aggregator/gen/go/feed/v1"
	"github.com/gin-gonic/gin"
	"github.com/nats-io/nats.go"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	// Connect to Feed Service
	feedServiceUrl := os.Getenv("FEED_SERVICE_URL")
	if feedServiceUrl == "" {
		feedServiceUrl = "localhost:50051"
	}
	conn, err := grpc.Dial(feedServiceUrl, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()
	feedClient := feedv1.NewFeedServiceClient(conn)

	// Connect to NATS
	natsUrl := os.Getenv("NATS_URL")
	if natsUrl == "" {
		natsUrl = nats.DefaultURL
	}
	nc, err := nats.Connect(natsUrl)
	if err != nil {
		log.Printf("Warning: Failed to connect to NATS: %v", err)
	} else {
		defer nc.Close()
	}

	r := gin.Default()

	// CORS
	r.Use(func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization")
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		c.Next()
	})

	// Auth Middleware
	authMiddleware := func(c *gin.Context) {
		tokenString := c.GetHeader("Authorization")
		if tokenString == "" {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "Missing Authorization header"})
			return
		}
		// In real app, validate JWT here. For now, just check presence.
		c.Next()
	}

	api := r.Group("/api")
	api.Use(authMiddleware)
	{
		api.GET("/feeds", func(c *gin.Context) {
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()

			res, err := feedClient.ListFeeds(ctx, &feedv1.ListFeedsRequest{})
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
			c.JSON(http.StatusOK, res.Feeds)
		})

		api.POST("/feeds", func(c *gin.Context) {
			var req struct {
				Url string `json:"url" binding:"required"`
			}
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()

			res, err := feedClient.CreateFeed(ctx, &feedv1.CreateFeedRequest{Url: req.Url})
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
			c.JSON(http.StatusCreated, res.Feed)
		})

		api.GET("/search", func(c *gin.Context) {
			query := c.Query("q")
			if query == "" {
				c.JSON(http.StatusBadRequest, gin.H{"error": "Missing query param 'q'"})
				return
			}
			
			// Proxy to Qdrant (Mock)
			// In real app: POST /collections/videos/points/search
			c.JSON(http.StatusOK, gin.H{
				"results": []gin.H{
					{"id": "1", "score": 0.9, "payload": gin.H{"title": "Mock Result for " + query}},
				},
			})
		})
	}

	// SSE Endpoint (Public for now, or move to /api)
	r.GET("/events", func(c *gin.Context) {
		c.Writer.Header().Set("Content-Type", "text/event-stream")
		c.Writer.Header().Set("Cache-Control", "no-cache")
		c.Writer.Header().Set("Connection", "keep-alive")
		c.Writer.Header().Set("Transfer-Encoding", "chunked")

		if nc == nil {
			c.SSEvent("error", "NATS not connected")
			return
		}

		// Subscribe to all video events
		sub, err := nc.SubscribeSync("video.>")
		if err != nil {
			c.SSEvent("error", "Failed to subscribe")
			return
		}
		defer sub.Unsubscribe()

		c.Stream(func(w io.Writer) bool {
			msg, err := sub.NextMsg(10 * time.Second)
			if err != nil {
				if err == nats.ErrTimeout {
					c.SSEvent("ping", "keepalive")
					return true
				}
				return false
			}
			c.SSEvent("message", string(msg.Data)) // In real app, decode proto and send JSON
			return true
		})
	})

	r.Run(":8080")
}
