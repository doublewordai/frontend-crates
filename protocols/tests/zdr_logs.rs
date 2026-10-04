// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol logs must never carry request text: zero-data-retention customers
//! require that nothing they send lands in shipped logs. Every request-derived
//! value below carries a sentinel, and the test fails if any captured event
//! (at TRACE) contains it.

use std::io::Write;
use std::sync::{Arc, Mutex};

use dynamo_protocols::types::anthropic::{AnthropicContentBlock, CacheControl};
use tracing_subscriber::fmt::MakeWriter;

const SENTINEL: &str = "ZDRSENTINEL";

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

struct CaptureWriter(Arc<Mutex<Vec<u8>>>);

impl Write for CaptureWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for Capture {
    type Writer = CaptureWriter;
    fn make_writer(&'a self) -> Self::Writer {
        CaptureWriter(self.0.clone())
    }
}

#[test]
fn protocol_logs_carry_no_request_text() {
    let capture = Capture::default();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_writer(capture.clone())
        .with_ansi(false)
        .without_time()
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        let cache_control = CacheControl {
            ttl: Some("ZDRSENTINEL ttl".to_string()),
            ..CacheControl::default()
        };
        assert_eq!(cache_control.ttl_seconds(), 300);

        let block: AnthropicContentBlock = serde_json::from_value(serde_json::json!({
            "type": "ZDRSENTINEL_block_type",
            "ZDRSENTINEL_field": "ZDRSENTINEL value",
        }))
        .unwrap();
        assert!(matches!(block, AnthropicContentBlock::Other(_)));
    });

    let logs = String::from_utf8_lossy(&capture.0.lock().unwrap()).into_owned();
    for message in [
        "Unrecognized TTL",
        "Unrecognized Anthropic content block type",
    ] {
        assert!(
            logs.contains(message),
            "site never fired: {message}\n{logs}"
        );
    }
    let leaks: Vec<&str> = logs
        .lines()
        .filter(|line| line.to_lowercase().contains(&SENTINEL.to_lowercase()))
        .collect();
    assert!(
        leaks.is_empty(),
        "log lines carried request text:\n{}",
        leaks.join("\n")
    );
}
