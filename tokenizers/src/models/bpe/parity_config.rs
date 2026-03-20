use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Deserialize a field that can be either a single string or a list of strings.
fn string_or_vec<'de, D>(deserializer: D) -> std::result::Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct StringOrVec;

    impl<'de> Visitor<'de> for StringOrVec {
        type Value = Vec<String>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a string or a list of strings")
        }

        fn visit_str<E: de::Error>(self, value: &str) -> std::result::Result<Vec<String>, E> {
            Ok(vec![value.to_owned()])
        }

        fn visit_string<E: de::Error>(self, value: String) -> std::result::Result<Vec<String>, E> {
            Ok(vec![value])
        }

        fn visit_seq<S: SeqAccess<'de>>(
            self,
            mut seq: S,
        ) -> std::result::Result<Vec<String>, S::Error> {
            let mut v = Vec::new();
            while let Some(s) = seq.next_element()? {
                v.push(s);
            }
            Ok(v)
        }
    }

    deserializer.deserialize_any(StringOrVec)
}

/// Deserialize an optional field that can be either a single string or a list of strings.
fn option_string_or_vec<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<Vec<String>>, D::Error>
where
    D: Deserializer<'de>,
{
    struct OptStringOrVec;

    impl<'de> Visitor<'de> for OptStringOrVec {
        type Value = Option<Vec<String>>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("null, a string, or a list of strings")
        }

        fn visit_none<E: de::Error>(self) -> std::result::Result<Option<Vec<String>>, E> {
            Ok(None)
        }

        fn visit_unit<E: de::Error>(self) -> std::result::Result<Option<Vec<String>>, E> {
            Ok(None)
        }

        fn visit_str<E: de::Error>(
            self,
            value: &str,
        ) -> std::result::Result<Option<Vec<String>>, E> {
            Ok(Some(vec![value.to_owned()]))
        }

        fn visit_string<E: de::Error>(
            self,
            value: String,
        ) -> std::result::Result<Option<Vec<String>>, E> {
            Ok(Some(vec![value]))
        }

        fn visit_seq<S: SeqAccess<'de>>(
            self,
            mut seq: S,
        ) -> std::result::Result<Option<Vec<String>>, S::Error> {
            let mut v = Vec::new();
            while let Some(s) = seq.next_element()? {
                v.push(s);
            }
            Ok(Some(v))
        }
    }

    deserializer.deserialize_any(OptStringOrVec)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageConfig {
    pub name: String,
    #[serde(deserialize_with = "string_or_vec")]
    pub input: Vec<String>,
    #[serde(
        default,
        deserialize_with = "option_string_or_vec",
        skip_serializing_if = "Option::is_none"
    )]
    pub dev: Option<Vec<String>>,
    #[serde(default)]
    pub ratio: Option<f64>,
    #[serde(default = "default_text_column")]
    pub text_column: String,
}

fn default_text_column() -> String {
    "text".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub languages: Vec<LanguageConfig>,
}

impl TrainingConfig {
    pub fn from_file(path: &str) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        Ok(serde_json::from_reader(reader)?)
    }

    pub fn ratios(&self) -> Vec<f64> {
        self.languages.iter().map(|l| l.ratio.unwrap_or(1.0)).collect()
    }

    /// Returns true if any language in the config has dev files specified.
    pub fn has_dev(&self) -> bool {
        self.languages
            .iter()
            .any(|l| l.dev.as_ref().is_some_and(|d| !d.is_empty()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_input() {
        let json = r#"{"languages": [{"name": "en", "input": "train.txt", "ratio": 1.0}]}"#;
        let config: TrainingConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.languages[0].input, vec!["train.txt"]);
        assert!(config.languages[0].dev.is_none());
    }

    #[test]
    fn test_vec_input() {
        let json =
            r#"{"languages": [{"name": "en", "input": ["a.txt", "b.txt"], "ratio": 1.0}]}"#;
        let config: TrainingConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.languages[0].input, vec!["a.txt", "b.txt"]);
    }

    #[test]
    fn test_dev_field() {
        let json = r#"{"languages": [{"name": "en", "input": "train.txt", "dev": "dev.txt", "ratio": 1.0}]}"#;
        let config: TrainingConfig = serde_json::from_str(json).unwrap();
        assert_eq!(
            config.languages[0].dev,
            Some(vec!["dev.txt".to_string()])
        );
        assert!(config.has_dev());
    }

    #[test]
    fn test_dev_field_vec() {
        let json = r#"{"languages": [{"name": "en", "input": "train.txt", "dev": ["d1.txt", "d2.txt"], "ratio": 1.0}]}"#;
        let config: TrainingConfig = serde_json::from_str(json).unwrap();
        assert_eq!(
            config.languages[0].dev,
            Some(vec!["d1.txt".to_string(), "d2.txt".to_string()])
        );
    }

    #[test]
    fn test_no_dev_field() {
        let json = r#"{"languages": [{"name": "en", "input": "train.txt", "ratio": 1.0}]}"#;
        let config: TrainingConfig = serde_json::from_str(json).unwrap();
        assert!(config.languages[0].dev.is_none());
        assert!(!config.has_dev());
    }

    #[test]
    fn test_text_column_default() {
        let json = r#"{"languages": [{"name": "en", "input": "train.txt", "ratio": 1.0}]}"#;
        let config: TrainingConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.languages[0].text_column, "text");
    }

    #[test]
    fn test_ratio_optional() {
        let json = r#"{"languages": [{"name": "en", "input": "train.txt"}]}"#;
        let config: TrainingConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.languages[0].ratio, None);
        assert_eq!(config.ratios(), vec![1.0]);
    }

    #[test]
    fn test_text_column_custom() {
        let json = r#"{"languages": [{"name": "code", "input": "code.parquet", "ratio": 1.0, "text_column": "content"}]}"#;
        let config: TrainingConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.languages[0].text_column, "content");
    }
}
