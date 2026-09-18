import time
import ast
import re
import logging
import os
from dataclasses import dataclass
from typing import List, Union, Optional
import requests
from tqdm import tqdm

@dataclass
class GroqConfig:
    """Configuration settings for Groq client"""
    model: str = "llama-3.3-70b-versatile"  # Groq's default model
    max_retries: int = 2
    initial_temperature: float = 0.5
    log_file: str = "./run_logs/groq_client_errors.log"

class ResponseValidationError(ValueError):
    """Custom exception for response validation errors"""
    pass

class GroqClient:
    def __init__(self, model: str, config: Optional[GroqConfig] = None):
        """
        Initialize Groq client with configuration
        
        Args:
            model: Model name to use
            config: GroqConfig object with client settings
        """
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not found")
            
        self.model = model
        self.config = config or GroqConfig()
        self.config.model = model
        self._setup_logging()
        self.retry_attempts = 0
        self.fallback_ids = None
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _setup_logging(self) -> None:
        """Configure logging settings"""
        logging.basicConfig(
            filename=self.config.log_file,
            level=logging.ERROR,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filemode="w"
        )

    def _get_temperature(self) -> float:
        """Get temperature"""
        return 0.5

    def _validate_response(
        self,
        response_list: List[int],
        min_val: int,
        max_val: int
    ) -> None:
        """
        Validate the response meets all requirements
        
        Args:
            response_list: List of integers to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Raises:
            ResponseValidationError: If validation fails
        """
        if not isinstance(response_list, list):
            raise ResponseValidationError("Response is not a list")
        
        if len(response_list) != 5:
            raise ResponseValidationError("Response list must contain exactly 5 elements")
        
        if not all(isinstance(x, (int, float)) and min_val <= x <= max_val for x in response_list):
            raise ResponseValidationError(
                f"Response list contains invalid values or values outside range [{min_val}, {max_val}]"
            )

    def _handle_validation_error(
        self,
        error: ResponseValidationError,
        response_list: List[int],
        index: int,
        min_val: int,
        max_val: int
    ) -> List[int]:
        """
        Handle different types of validation errors and apply fixes
        
        Args:
            error: The validation error
            response_list: Current response list
            index: Index in the input sequence
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            List[int]: Fixed response list
        """
        if "Response is not a list" in str(error):
            return self.fallback_ids[index]

        if any(msg in str(error) for msg in ['exactly 5']):
            unique_values = list(set(response_list))
            return self._complete_list_with_fallbacks(unique_values, index)

        if 'allowed range' in str(error):
            clamped_values = list(set(min(max(min_val, x), max_val) for x in response_list))
            return self._complete_list_with_fallbacks(clamped_values, index)

        return self.fallback_ids[index]

    def _complete_list_with_fallbacks(self, current_list: List[int], index: int) -> List[int]:
        """Complete a partial list using fallback IDs"""
        result = current_list.copy()
        for fallback_id in self.fallback_ids[index]:
            if fallback_id not in result:
                result.append(fallback_id)
                if len(result) == 5:
                    break
        return result

    def _call_api(self, prompt: str, temperature: float) -> List[int]:
        """Make the API call to Groq"""
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": "You are a helpful table ranker."},
                {"role": "user", "content": prompt}],
            "temperature": temperature
        }

        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"API call failed with status {response.status_code}: {response.text}")

        content = response.json()["choices"][0]["message"]["content"]
        
        response_list = re.search(r'\[.*?\]', content)
        if response_list:
            response_list = response_list.group(0)
        
        return ast.literal_eval(response_list.strip())

    def process_single_prompt(
        self,
        index: int,
        prompt: str,
        temperature: Optional[float] = None,
        min_val: int = 0,
        max_val: int = 728
    ) -> List[int]:
        """
        Process a single prompt and return validated response
        
        Args:
            index: Index in the sequence of prompts
            prompt: The input prompt
            temperature: Optional temperature override
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            List[int]: List of 5 unique integers within range
        """
        temperature = temperature if temperature is not None else self._get_temperature()

        try:
            response_list = self._call_api(prompt, temperature)
            self._validate_response(response_list, min_val, max_val)
            self.retry_attempts = 0
            return response_list

        except Exception as e:
            logging.error(f"Exception: {e}. Retry attempt {self.retry_attempts + 1}")
            
            if self.retry_attempts >= self.config.max_retries:
                self.retry_attempts = 0
                if isinstance(e, ResponseValidationError):
                    return self._handle_validation_error(e, response_list, index, min_val, max_val)
                logging.error(f"Max retries reached. Using fallback ids: {self.fallback_ids[index]}")
                return self.fallback_ids[index]

            retry_time = 1  # Groq doesn't provide retry-after headers, using fixed delay
            time.sleep(retry_time)
            
            self.retry_attempts += 1
            return self.process_single_prompt(index, prompt, temperature, min_val, max_val)

    def process_prompts(
        self,
        inputs: Union[str, List[str]],
        temperature: Optional[float] = None,
        min_val: int = 0,
        max_val: int = 728,
        fallback_ids: Optional[List[List[int]]] = None
    ) -> List[List[int]]:
        """
        Process multiple prompts and return responses
        
        Args:
            inputs: Single input string or list of input strings
            temperature: Optional temperature override
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            fallback_ids: Fallback values for each input
            
        Returns:
            List[List[int]]: List of response lists
        """
        self.fallback_ids = fallback_ids
        if isinstance(inputs, str):
            inputs = [inputs]

        print(f"Processing {len(inputs)} prompts...")
        return [
            self.process_single_prompt(i, prompt, temperature, min_val, max_val)
            for i, prompt in tqdm(enumerate(inputs), total=len(inputs))
        ]