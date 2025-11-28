package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"time"

	feedv1 "github.com/arietids/video-rss-aggregator/gen/go/feed/v1"
	"github.com/jackc/pgx/v5/pgxpool"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/types/known/timestamppb"
)

type server struct {
	feedv1.UnimplementedFeedServiceServer
	db *pgxpool.Pool
}

func (s *server) CreateFeed(ctx context.Context, req *feedv1.CreateFeedRequest) (*feedv1.CreateFeedResponse, error) {
	var id string
	var createdAt, updatedAt time.Time

	// Simple insert, assuming URL is unique (handled by DB constraint)
	err := s.db.QueryRow(ctx, 
		"INSERT INTO feeds (url) VALUES ($1) RETURNING id, created_at, updated_at", 
		req.Url).Scan(&id, &createdAt, &updatedAt)
	
	if err != nil {
		return nil, fmt.Errorf("failed to insert feed: %v", err)
	}

	return &feedv1.CreateFeedResponse{
		Feed: &feedv1.Feed{
			Id:        id,
			Url:       req.Url,
			Status:    "ACTIVE",
			CreatedAt: timestamppb.New(createdAt),
			UpdatedAt: timestamppb.New(updatedAt),
		},
	}, nil
}

func (s *server) ListFeeds(ctx context.Context, req *feedv1.ListFeedsRequest) (*feedv1.ListFeedsResponse, error) {
	rows, err := s.db.Query(ctx, "SELECT id, url, title, description, status, created_at, updated_at FROM feeds LIMIT 50")
	if err != nil {
		return nil, fmt.Errorf("failed to query feeds: %v", err)
	}
	defer rows.Close()

	var feeds []*feedv1.Feed
	for rows.Next() {
		var f feedv1.Feed
		var createdAt, updatedAt time.Time
		var title, description *string
		
		if err := rows.Scan(&f.Id, &f.Url, &title, &description, &f.Status, &createdAt, &updatedAt); err != nil {
			return nil, fmt.Errorf("failed to scan row: %v", err)
		}
		
		if title != nil { f.Title = *title }
		if description != nil { f.Description = *description }
		f.CreatedAt = timestamppb.New(createdAt)
		f.UpdatedAt = timestamppb.New(updatedAt)
		
		feeds = append(feeds, &f)
	}

	return &feedv1.ListFeedsResponse{
		Feeds: feeds,
	}, nil
}

func main() {
	dbUrl := os.Getenv("DATABASE_URL")
	if dbUrl == "" {
		dbUrl = "postgres://user:password@localhost:5432/video_rss"
	}

	dbPool, err := pgxpool.New(context.Background(), dbUrl)
	if err != nil {
		log.Fatalf("Unable to connect to database: %v\n", err)
	}
	defer dbPool.Close()

	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	s := grpc.NewServer()
	feedv1.RegisterFeedServiceServer(s, &server{db: dbPool})
	
	log.Printf("server listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
