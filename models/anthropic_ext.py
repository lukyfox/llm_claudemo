from anthropic import Anthropic

class AnthropicExt(Anthropic):
    """
    Inherit Anthropic class to extend with custom functions
    """

    def _get_latest_model_id(self, model_name: str) -> str:
        """
        Get latest model id from Anthropic API searched by model name - models are pre-ordered by release date
        :param model_name: simple name of the model to search for (i.e. enter haiku to find latest
        claude-haiku-4-5-20251001)
        :return: model id as a string (i.e. claude-haiku-4-5-20251001)
        """
        models = self.models.list()
        placeholder = ''
        if not models:
            raise RuntimeError("No models found")
        for model in models:
            if not placeholder and "sonnet" in model.id.lower():
                placeholder = model.id
            if model_name.lower() in model.id.lower():
                return model.id
        # when no match then return first found Sonnet model, or raise if none available
        if placeholder:
            return placeholder
        raise RuntimeError(f"No model matching '{model_name}' found and no Sonnet fallback available.")

    def _get_all_model_ids(self) -> set[str]:
        """
        Get all model ids from Anthropic API
        :return: all model ids as a set of strings, i.e. (claude-haiku-4-5-20251001, claude-sonnet-4-6,...)
        """
        models = self.models.list()
        if not models:
            raise RuntimeError("No models found")
        return {model.id for model in models}

    def get_fitting_model_id(self, description: str) -> str:
        """
        Get model id from Anthropic API searched by description - either simply by model name
        or try to find exact model ID (if given i.e. claude-haiku-4-5-20251001 as argument) or by message analysis.
        :param description: question (or it may be a snippet of communication history), or model name (i.e. Haiku),
                            or exact model id (i.e. claude-haiku-4-5-20251001) - in case of model id when no match
                            is found then continue with attempt to get best fitting model ID by LLM
        :return: model id as a string (i.e. claude-haiku-4-5-20251001)
        """
        if description:
            if description.lower() in ('haiku', 'sonnet', 'opus'):
                return self._get_latest_model_id(description.lower())
            elif model_candidate := [
                item for item in description.lower().split(' ')
                if any(name in item for name in ['haiku', 'sonnet', 'opus'])
            ]:
                for model_id in self._get_all_model_ids():
                    if model_candidate[0] in model_id.lower():
                        return model_id

        system_message = ('Act as a specialist for Anthropic Claude models who is able to provide advice '
                          'which model best fits to given requirements. For example, when a simple question '
                          'without deep reasoning or research is needed to process, select Haiku as the most suitable '
                          'model, or when the question requires complex area with a need of research, '
                          'multi-turn conversation and citations, then return Opus as the most suitable. '
                          'Always return only the base name of the model - either Haiku, or Sonnet or Opus. '
                          'Return Sonnet if you cannot decide.')
        messages = [
            {'role': 'user', 'content': description},
            {'role': 'assistant', 'content': '```'}
        ]

        response = self.messages.create(
            model=self._get_latest_model_id('haiku'),
            max_tokens=100,
            system=system_message,
            messages=messages,
            stop_sequences=['```']
        )

        return self._get_latest_model_id(response.content[0].text.strip())
