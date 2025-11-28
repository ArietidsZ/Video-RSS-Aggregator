fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(false)
        .build_client(true)
        .compile(
            &["../proto/feed.proto", "../proto/events.proto"],
            &["../proto"],
        )?;
    Ok(())
}
