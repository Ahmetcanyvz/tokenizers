use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageConfig {
    pub name: String,
    pub input: Vec<String>,
    pub ratio: f64,
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
        self.languages.iter().map(|l| l.ratio).collect()
    }
}
