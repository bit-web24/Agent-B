//! LLM response caching — avoid duplicate API calls by caching
//! responses keyed by SHA-256 hash of the messages payload.

use crate::types::LlmResponse;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────────────
// Cache key helper
// ─────────────────────────────────────────────────────────────────────────────

/// Computes a SHA-256 hex digest from an LLM message payload.
pub fn cache_key(messages: &[Value], model: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(model.as_bytes());
    for msg in messages {
        hasher.update(msg.to_string().as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

// ─────────────────────────────────────────────────────────────────────────────
// Stats
// ─────────────────────────────────────────────────────────────────────────────

/// Observable cache performance stats.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Trait
// ─────────────────────────────────────────────────────────────────────────────

/// Pluggable LLM response cache.
pub trait LlmCache: Send + Sync {
    /// Look up a cached response by key. Returns `None` on miss or expiry.
    fn get(&self, key: &str) -> Option<LlmResponse>;

    /// Insert a response keyed by SHA-256 hex digest.
    fn put(&self, key: String, response: LlmResponse);

    /// Return current cache stats.
    fn stats(&self) -> CacheStats;
}

// ─────────────────────────────────────────────────────────────────────────────
// NoopCache
// ─────────────────────────────────────────────────────────────────────────────

/// Does nothing — used when caching is disabled.
pub struct NoopCache;

impl LlmCache for NoopCache {
    fn get(&self, _key: &str) -> Option<LlmResponse> {
        None
    }
    fn put(&self, _key: String, _response: LlmResponse) {}
    fn stats(&self) -> CacheStats {
        CacheStats::default()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// InMemoryCache
// ─────────────────────────────────────────────────────────────────────────────

struct CacheEntry {
    response: LlmResponse,
    inserted: Instant,
    last_used: Instant,
}

/// Thread-safe in-memory LRU cache with TTL expiration.
pub struct InMemoryCache {
    inner: Mutex<InMemoryCacheInner>,
    ttl: Duration,
    max_size: usize,
}

struct InMemoryCacheInner {
    entries: HashMap<String, CacheEntry>,
    stats: CacheStats,
}

impl InMemoryCache {
    /// Create a new in-memory cache.
    ///
    /// * `max_size` — maximum number of entries (0 = unlimited)
    /// * `ttl`      — entries older than this are treated as expired
    pub fn new(max_size: usize, ttl: Duration) -> Self {
        Self {
            inner: Mutex::new(InMemoryCacheInner {
                entries: HashMap::new(),
                stats: CacheStats::default(),
            }),
            ttl,
            max_size,
        }
    }

    /// Evict the least-recently-used entry.
    fn evict_lru(inner: &mut InMemoryCacheInner) {
        if let Some(oldest_key) = inner
            .entries
            .iter()
            .min_by_key(|(_, e)| e.last_used)
            .map(|(k, _)| k.clone())
        {
            inner.entries.remove(&oldest_key);
            inner.stats.evictions += 1;
        }
    }
}

impl LlmCache for InMemoryCache {
    fn get(&self, key: &str) -> Option<LlmResponse> {
        let mut inner = self.inner.lock().unwrap();

        // First check expiry — if expired, remove and count as eviction
        let expired = inner
            .entries
            .get(key)
            .map(|e| e.inserted.elapsed() >= self.ttl)
            .unwrap_or(false);

        if expired {
            inner.entries.remove(key);
            inner.stats.evictions += 1;
            inner.stats.misses += 1;
            return None;
        }

        // Now try to get a valid entry
        if let Some(entry) = inner.entries.get_mut(key) {
            entry.last_used = Instant::now();
            let result = entry.response.clone();
            inner.stats.hits += 1;
            return Some(result);
        }

        inner.stats.misses += 1;
        None
    }

    fn put(&self, key: String, response: LlmResponse) {
        let mut inner = self.inner.lock().unwrap();

        // If we're at capacity, evict LRU
        if self.max_size > 0
            && inner.entries.len() >= self.max_size
            && !inner.entries.contains_key(&key)
        {
            Self::evict_lru(&mut inner);
        }

        inner.entries.insert(
            key,
            CacheEntry {
                response,
                inserted: Instant::now(),
                last_used: Instant::now(),
            },
        );
    }

    fn stats(&self) -> CacheStats {
        self.inner.lock().unwrap().stats.clone()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ToolCall;
    use std::collections::HashMap;

    fn make_response(text: &str) -> LlmResponse {
        LlmResponse::FinalAnswer {
            content: text.to_string(),
            usage: None,
        }
    }

    fn make_tool_response(name: &str) -> LlmResponse {
        LlmResponse::ToolCall {
            tool: ToolCall {
                name: name.to_string(),
                args: HashMap::new(),
                id: None,
            },
            confidence: 0.9,
            usage: None,
        }
    }

    #[test]
    fn test_cache_key_deterministic() {
        let msgs = vec![serde_json::json!({"role": "user", "content": "hello"})];
        let k1 = cache_key(&msgs, "gpt-4");
        let k2 = cache_key(&msgs, "gpt-4");
        assert_eq!(k1, k2);
        assert_eq!(k1.len(), 64); // SHA-256 hex = 64 chars
    }

    #[test]
    fn test_cache_key_model_matters() {
        let msgs = vec![serde_json::json!({"role": "user", "content": "hello"})];
        let k1 = cache_key(&msgs, "gpt-4");
        let k2 = cache_key(&msgs, "gpt-3.5-turbo");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_noop_cache() {
        let cache = NoopCache;
        cache.put("k".to_string(), make_response("test"));
        assert!(cache.get("k").is_none());
        assert_eq!(cache.stats().hits, 0);
    }

    #[test]
    fn test_in_memory_hit_and_miss() {
        let cache = InMemoryCache::new(100, Duration::from_secs(60));
        assert!(cache.get("k").is_none());
        assert_eq!(cache.stats().misses, 1);

        cache.put("k".to_string(), make_response("hello"));
        let resp = cache.get("k");
        assert!(resp.is_some());
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_in_memory_ttl_expiry() {
        let cache = InMemoryCache::new(100, Duration::from_millis(10));
        cache.put("k".to_string(), make_response("hello"));
        std::thread::sleep(Duration::from_millis(20));
        assert!(cache.get("k").is_none());
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_in_memory_lru_eviction() {
        let cache = InMemoryCache::new(2, Duration::from_secs(60));
        cache.put("a".to_string(), make_response("a"));
        cache.put("b".to_string(), make_response("b"));

        // Access "a" so "b" becomes LRU
        let _ = cache.get("a");

        // Adding "c" should evict "b" (LRU)
        cache.put("c".to_string(), make_response("c"));

        assert!(cache.get("a").is_some());
        assert!(cache.get("b").is_none()); // evicted
        assert!(cache.get("c").is_some());
    }

    #[test]
    fn test_cache_stats_hit_rate() {
        let mut stats = CacheStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
        stats.hits = 3;
        stats.misses = 1;
        assert!((stats.hit_rate() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_cache_preserves_tool_calls() {
        let cache = InMemoryCache::new(100, Duration::from_secs(60));
        let resp = make_tool_response("search");
        cache.put("k".to_string(), resp.clone());
        let cached = cache.get("k").unwrap();
        match cached {
            LlmResponse::ToolCall { tool, .. } => assert_eq!(tool.name, "search"),
            _ => panic!("Wrong response type"),
        }
    }
}
