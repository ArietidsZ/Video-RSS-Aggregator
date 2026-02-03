use anyhow::Result;
use chrono::{DateTime, Utc};
use rss::{ChannelBuilder, GuidBuilder, ItemBuilder};

#[derive(Clone, Debug)]
pub struct RssItemData {
    pub title: String,
    pub link: String,
    pub summary: String,
    pub key_points: Vec<String>,
    pub published_at: Option<DateTime<Utc>>,
    pub guid: Option<String>,
}

pub fn render_feed(
    title: &str,
    link: &str,
    description: &str,
    items: &[RssItemData],
) -> Result<String> {
    let mut rss_items = Vec::with_capacity(items.len());

    for item in items {
        let mut description_text = item.summary.clone();
        if !item.key_points.is_empty() {
            let bullets = item
                .key_points
                .iter()
                .map(|point| format!("- {}", point))
                .collect::<Vec<_>>()
                .join("\n");
            description_text = format!("{}\n\n{}", description_text, bullets);
        }

        let guid = item
            .guid
            .as_ref()
            .map(|value| GuidBuilder::default().value(value).permalink(false).build());

        let rss_item = ItemBuilder::default()
            .title(Some(item.title.clone()))
            .link(Some(item.link.clone()))
            .description(Some(description_text))
            .guid(guid)
            .pub_date(item.published_at.map(|dt| dt.to_rfc2822()))
            .build();

        rss_items.push(rss_item);
    }

    let channel = ChannelBuilder::default()
        .title(title)
        .link(link)
        .description(description)
        .items(rss_items)
        .build();

    Ok(channel.to_string())
}
