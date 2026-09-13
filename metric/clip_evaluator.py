# Evaluates CLIP Score between text prompts and images using CLIP ViT-B/32 model.

import torch
from PIL import Image
import os
import clip


class CLIPScoreEvaluator:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()
    
    def calculate_clip_score(self, image_path: str, text_prompt: str) -> float:
        """
        Calculate CLIP score between an image and a text prompt.
        
        Args:
            image_path: Path to the image file
            text_prompt: Text description to compare
            
        Returns:
            CLIP similarity score (0-100)
        """
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        text_input = clip.tokenize([text_prompt]).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate cosine similarity
            similarity = (image_features @ text_features.T).squeeze()
            
            # Convert to score (0-100)
            clip_score = similarity.item() * 100
        
        return clip_score


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate CLIP score between image and text prompt")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to compare")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    evaluator = CLIPScoreEvaluator(device=args.device)
    
    print(f"\nCalculating CLIP score...")
    print(f"Image: {args.image}")
    print(f"Prompt: {args.prompt}")
    print("-" * 50)
    
    score = evaluator.calculate_clip_score(args.image, args.prompt)
    
    print(f"\nCLIP Score: {score:.2f}/100")
    
if __name__ == "__main__":
    main()
