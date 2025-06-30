from utils import read_yaml_config

yaml_config = read_yaml_config("./config.yml")
asr_type = yaml_config.get("asr_type", "whisper")  # Default to "whisper" if not specified
if asr_type not in ["whisper", "nemo"]:
    raise ValueError(f"❌ Invalid ASR type specified: {asr_type}. Supported types are 'whisper' and 'nemo'.")
print(f"🔧 ASR Type set to: {asr_type}"
      )

