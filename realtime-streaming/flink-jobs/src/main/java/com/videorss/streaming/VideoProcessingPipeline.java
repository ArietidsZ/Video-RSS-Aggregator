package com.videorss.streaming;

import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.time.Time;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.kafka.sink.KafkaRecordSerializationSchema;
import org.apache.flink.connector.kafka.sink.KafkaSink;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.streaming.api.CheckpointingMode;
import org.apache.flink.streaming.api.datastream.AsyncDataStream;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.async.ResultFuture;
import org.apache.flink.streaming.api.functions.async.RichAsyncFunction;
import org.apache.flink.streaming.api.functions.windowing.ProcessWindowFunction;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.util.Collector;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.time.Duration;
import java.util.Collections;
import java.util.Properties;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/**
 * Main Flink job for real-time video processing pipeline
 */
public class VideoProcessingPipeline {

    private static final String KAFKA_BOOTSTRAP_SERVERS = "kafka-1:29092,kafka-2:29093,kafka-3:29094";
    private static final String INPUT_TOPIC = "video-events";
    private static final String TRANSCRIPTION_TOPIC = "transcription-results";
    private static final String SUMMARY_TOPIC = "summary-results";
    private static final String RSS_TOPIC = "rss-updates";

    public static void main(String[] args) throws Exception {
        // Set up the execution environment
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // Configure checkpointing
        env.enableCheckpointing(30000); // 30 seconds
        env.getCheckpointConfig().setCheckpointingMode(CheckpointingMode.EXACTLY_ONCE);
        env.getCheckpointConfig().setMinPauseBetweenCheckpoints(10000);
        env.getCheckpointConfig().setCheckpointTimeout(60000);
        env.getCheckpointConfig().setMaxConcurrentCheckpoints(1);

        // Configure restart strategy
        env.setRestartStrategy(RestartStrategies.fixedDelayRestart(
            3,  // number of restart attempts
            Time.of(10, TimeUnit.SECONDS)  // delay between restarts
        ));

        // Set parallelism
        env.setParallelism(4);

        // Create Kafka source for video events
        KafkaSource<String> videoSource = KafkaSource.<String>builder()
            .setBootstrapServers(KAFKA_BOOTSTRAP_SERVERS)
            .setTopics(INPUT_TOPIC)
            .setGroupId("video-processing-group")
            .setStartingOffsets(OffsetsInitializer.earliest())
            .setValueOnlyDeserializer(new SimpleStringSchema())
            .build();

        // Read video events stream
        DataStream<String> videoEvents = env.fromSource(
            videoSource,
            WatermarkStrategy.<String>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                .withIdleness(Duration.ofSeconds(60)),
            "Video Events Source"
        );

        // Parse video events
        DataStream<VideoEvent> parsedEvents = videoEvents
            .map(new VideoEventParser())
            .name("Parse Video Events");

        // Filter valid events
        DataStream<VideoEvent> validEvents = parsedEvents
            .filter(event -> event != null && event.isValid())
            .name("Filter Valid Events");

        // Process video metadata extraction (async)
        DataStream<VideoMetadata> videoMetadata = AsyncDataStream.unorderedWait(
            validEvents,
            new AsyncMetadataExtractor(),
            60,  // timeout
            TimeUnit.SECONDS,
            100  // capacity
        ).name("Extract Video Metadata");

        // Branch stream for different processing paths
        DataStream<VideoMetadata> audioProcessing = videoMetadata
            .filter(metadata -> metadata.hasAudio())
            .name("Filter Audio");

        DataStream<VideoMetadata> visualProcessing = videoMetadata
            .filter(metadata -> metadata.hasVideo())
            .name("Filter Video");

        // Audio processing branch - Transcription
        DataStream<TranscriptionResult> transcriptions = AsyncDataStream.orderedWait(
            audioProcessing,
            new AsyncTranscriptionProcessor(),
            120,  // 2 minutes timeout for transcription
            TimeUnit.SECONDS,
            50
        ).name("Process Transcription");

        // Write transcription results to Kafka
        KafkaSink<String> transcriptionSink = KafkaSink.<String>builder()
            .setBootstrapServers(KAFKA_BOOTSTRAP_SERVERS)
            .setRecordSerializer(KafkaRecordSerializationSchema.builder()
                .setTopic(TRANSCRIPTION_TOPIC)
                .setValueSerializationSchema(new SimpleStringSchema())
                .build()
            )
            .build();

        transcriptions
            .map(new TranscriptionSerializer())
            .sinkTo(transcriptionSink)
            .name("Transcription Sink");

        // Summarization branch
        DataStream<SummaryResult> summaries = transcriptions
            .keyBy(t -> t.videoId)
            .window(TumblingEventTimeWindows.of(Time.seconds(30)))
            .process(new SummarizationWindowFunction())
            .name("Generate Summaries");

        // Write summaries to Kafka
        KafkaSink<String> summarySink = KafkaSink.<String>builder()
            .setBootstrapServers(KAFKA_BOOTSTRAP_SERVERS)
            .setRecordSerializer(KafkaRecordSerializationSchema.builder()
                .setTopic(SUMMARY_TOPIC)
                .setValueSerializationSchema(new SimpleStringSchema())
                .build()
            )
            .build();

        summaries
            .map(new SummarySerializer())
            .sinkTo(summarySink)
            .name("Summary Sink");

        // RSS generation branch
        DataStream<RSSUpdate> rssUpdates = summaries
            .flatMap(new RSSGenerator())
            .name("Generate RSS Updates");

        // Write RSS updates to Kafka
        KafkaSink<String> rssSink = KafkaSink.<String>builder()
            .setBootstrapServers(KAFKA_BOOTSTRAP_SERVERS)
            .setRecordSerializer(KafkaRecordSerializationSchema.builder()
                .setTopic(RSS_TOPIC)
                .setValueSerializationSchema(new SimpleStringSchema())
                .build()
            )
            .build();

        rssUpdates
            .map(new RSSSerializer())
            .sinkTo(rssSink)
            .name("RSS Sink");

        // Metrics and monitoring
        DataStream<ProcessingMetric> metrics = validEvents
            .flatMap(new MetricsCollector())
            .name("Collect Metrics");

        // Execute the job
        env.execute("Video Processing Pipeline");
    }

    /**
     * Parse raw video events
     */
    public static class VideoEventParser implements MapFunction<String, VideoEvent> {
        private final ObjectMapper mapper = new ObjectMapper();

        @Override
        public VideoEvent map(String value) throws Exception {
            try {
                return mapper.readValue(value, VideoEvent.class);
            } catch (Exception e) {
                System.err.println("Failed to parse event: " + e.getMessage());
                return null;
            }
        }
    }

    /**
     * Async metadata extraction
     */
    public static class AsyncMetadataExtractor extends RichAsyncFunction<VideoEvent, VideoMetadata> {
        private transient MetadataExtractorClient client;

        @Override
        public void open(Configuration parameters) throws Exception {
            client = new MetadataExtractorClient();
        }

        @Override
        public void asyncInvoke(VideoEvent event, ResultFuture<VideoMetadata> resultFuture) {
            CompletableFuture.supplyAsync(() -> {
                try {
                    return client.extractMetadata(event.url);
                } catch (Exception e) {
                    return new VideoMetadata(event.videoId, event.url, "error");
                }
            }).thenAccept(metadata -> {
                resultFuture.complete(Collections.singleton(metadata));
            });
        }
    }

    /**
     * Async transcription processing
     */
    public static class AsyncTranscriptionProcessor extends RichAsyncFunction<VideoMetadata, TranscriptionResult> {
        private transient TranscriptionClient client;

        @Override
        public void open(Configuration parameters) throws Exception {
            client = new TranscriptionClient("http://triton-server:8000");
        }

        @Override
        public void asyncInvoke(VideoMetadata metadata, ResultFuture<TranscriptionResult> resultFuture) {
            CompletableFuture.supplyAsync(() -> {
                try {
                    return client.transcribe(metadata);
                } catch (Exception e) {
                    return new TranscriptionResult(metadata.videoId, "", 0.0f);
                }
            }).thenAccept(result -> {
                resultFuture.complete(Collections.singleton(result));
            });
        }

        @Override
        public void timeout(VideoMetadata metadata, ResultFuture<TranscriptionResult> resultFuture) {
            resultFuture.complete(Collections.singleton(
                new TranscriptionResult(metadata.videoId, "TIMEOUT", 0.0f)
            ));
        }
    }

    /**
     * Window function for summarization
     */
    public static class SummarizationWindowFunction
            extends ProcessWindowFunction<TranscriptionResult, SummaryResult, String, TimeWindow> {

        @Override
        public void process(String key,
                          Context context,
                          Iterable<TranscriptionResult> elements,
                          Collector<SummaryResult> out) {
            StringBuilder fullTranscript = new StringBuilder();
            float totalConfidence = 0;
            int count = 0;

            for (TranscriptionResult result : elements) {
                fullTranscript.append(result.text).append(" ");
                totalConfidence += result.confidence;
                count++;
            }

            if (count > 0) {
                // Call summarization service
                String summary = generateSummary(fullTranscript.toString());
                float avgConfidence = totalConfidence / count;

                out.collect(new SummaryResult(
                    key,
                    summary,
                    fullTranscript.toString(),
                    avgConfidence,
                    context.window().getEnd()
                ));
            }
        }

        private String generateSummary(String text) {
            // Simplified - would call actual summarization service
            if (text.length() > 500) {
                return text.substring(0, 500) + "...";
            }
            return text;
        }
    }

    /**
     * RSS feed generator
     */
    public static class RSSGenerator extends RichFlatMapFunction<SummaryResult, RSSUpdate> {
        private ValueState<String> lastSummaryState;

        @Override
        public void open(Configuration config) {
            ValueStateDescriptor<String> descriptor = new ValueStateDescriptor<>(
                "lastSummary",
                TypeInformation.of(String.class)
            );
            lastSummaryState = getRuntimeContext().getState(descriptor);
        }

        @Override
        public void flatMap(SummaryResult summary, Collector<RSSUpdate> out) throws Exception {
            String lastSummary = lastSummaryState.value();

            // Check if summary is different from last one
            if (lastSummary == null || !lastSummary.equals(summary.summary)) {
                RSSUpdate update = new RSSUpdate(
                    summary.videoId,
                    summary.summary,
                    summary.windowEnd,
                    generateRSSContent(summary)
                );
                out.collect(update);
                lastSummaryState.update(summary.summary);
            }
        }

        private String generateRSSContent(SummaryResult summary) {
            return String.format(
                "<item><title>Video %s Summary</title><description>%s</description></item>",
                summary.videoId,
                summary.summary
            );
        }
    }

    /**
     * Metrics collector for monitoring
     */
    public static class MetricsCollector extends RichFlatMapFunction<VideoEvent, ProcessingMetric> {
        private transient long eventCount;
        private transient long lastReportTime;

        @Override
        public void open(Configuration config) {
            eventCount = 0;
            lastReportTime = System.currentTimeMillis();
        }

        @Override
        public void flatMap(VideoEvent event, Collector<ProcessingMetric> out) {
            eventCount++;

            long currentTime = System.currentTimeMillis();
            if (currentTime - lastReportTime > 10000) {  // Report every 10 seconds
                double throughput = eventCount * 1000.0 / (currentTime - lastReportTime);

                out.collect(new ProcessingMetric(
                    "throughput",
                    throughput,
                    currentTime
                ));

                eventCount = 0;
                lastReportTime = currentTime;
            }
        }
    }

    // Serializers
    public static class TranscriptionSerializer implements MapFunction<TranscriptionResult, String> {
        private final ObjectMapper mapper = new ObjectMapper();

        @Override
        public String map(TranscriptionResult value) throws Exception {
            return mapper.writeValueAsString(value);
        }
    }

    public static class SummarySerializer implements MapFunction<SummaryResult, String> {
        private final ObjectMapper mapper = new ObjectMapper();

        @Override
        public String map(SummaryResult value) throws Exception {
            return mapper.writeValueAsString(value);
        }
    }

    public static class RSSSerializer implements MapFunction<RSSUpdate, String> {
        private final ObjectMapper mapper = new ObjectMapper();

        @Override
        public String map(RSSUpdate value) throws Exception {
            return mapper.writeValueAsString(value);
        }
    }
}

// Data models (would typically be in separate files)
class VideoEvent {
    public String videoId;
    public String url;
    public String platform;
    public long timestamp;

    public boolean isValid() {
        return videoId != null && url != null;
    }
}

class VideoMetadata {
    public String videoId;
    public String url;
    public String status;
    public int duration;
    public boolean hasAudio;
    public boolean hasVideo;

    public VideoMetadata(String videoId, String url, String status) {
        this.videoId = videoId;
        this.url = url;
        this.status = status;
        this.hasAudio = true;
        this.hasVideo = true;
    }

    public boolean hasAudio() { return hasAudio; }
    public boolean hasVideo() { return hasVideo; }
}

class TranscriptionResult {
    public String videoId;
    public String text;
    public float confidence;

    public TranscriptionResult(String videoId, String text, float confidence) {
        this.videoId = videoId;
        this.text = text;
        this.confidence = confidence;
    }
}

class SummaryResult {
    public String videoId;
    public String summary;
    public String fullText;
    public float confidence;
    public long windowEnd;

    public SummaryResult(String videoId, String summary, String fullText, float confidence, long windowEnd) {
        this.videoId = videoId;
        this.summary = summary;
        this.fullText = fullText;
        this.confidence = confidence;
        this.windowEnd = windowEnd;
    }
}

class RSSUpdate {
    public String videoId;
    public String summary;
    public long timestamp;
    public String rssContent;

    public RSSUpdate(String videoId, String summary, long timestamp, String rssContent) {
        this.videoId = videoId;
        this.summary = summary;
        this.timestamp = timestamp;
        this.rssContent = rssContent;
    }
}

class ProcessingMetric {
    public String name;
    public double value;
    public long timestamp;

    public ProcessingMetric(String name, double value, long timestamp) {
        this.name = name;
        this.value = value;
        this.timestamp = timestamp;
    }
}

// Client stubs (would be actual implementations)
class MetadataExtractorClient {
    public VideoMetadata extractMetadata(String url) {
        // Stub implementation
        return new VideoMetadata("video123", url, "success");
    }
}

class TranscriptionClient {
    private String endpoint;

    public TranscriptionClient(String endpoint) {
        this.endpoint = endpoint;
    }

    public TranscriptionResult transcribe(VideoMetadata metadata) {
        // Stub implementation
        return new TranscriptionResult(metadata.videoId, "Sample transcription", 0.95f);
    }
}