import yaml
from string import Template
from functools import lru_cache
from geomas.core.utils import CONFIG_PATH
class PromptManager:
    def __init__(self, config_path=f"{CONFIG_PATH}/prompts/vision.yaml"):
        self.config_path = config_path
        self.prompts = self._load_yaml()

    @staticmethod
    def _load_yaml_from_file(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @lru_cache()
    def _load_yaml(self):
        data = self._load_yaml_from_file(self.config_path)
        if "prompts" not in data:
            raise ValueError("The prompt.yaml file must contain a 'prompts' field.")
        return data["prompts"]

    def get_roles(self):
        """Return all role names"""
        return list(self.prompts.keys())

    def get_prompt(self, role):
        """Returns the prompt content for the specified character (without rendering)"""
        if role not in self.prompts:
            raise KeyError(f"Not found prompt of role: {role}")
        return self.prompts[role]

    def render(self, role, **kwargs):
        """
        Renders a prompt for a specific character.
        Supports the {{var}} style (automatically converted to $var).
        """
        if role not in self.prompts:
            raise KeyError(f"Not found prompt of role: {role}")

        item = self.prompts[role]
        rendered = {}

        for key, text in item.items():
            # 支持 {{var}} -> $var
            for k in kwargs:
                text = text.replace(f"{{{{{k}}}}}", f"${k}")
            template = Template(text)
            rendered[key] = template.safe_substitute(**kwargs)

        return rendered

    def reload(self):
        """Reload YAML (suitable for hot reloading)"""
        self._load_yaml.cache_clear()
        self.prompts = self._load_yaml()
