---
title: "Chapter 17: Vision-Language Models"
sidebar_label: "Chapter 17: Vision-Language Models"
sidebar_position: 17
---

# Chapter 17: Vision-Language Models

## Advanced Vision-Language Integration for Humanoid Robotics

Welcome to Chapter 17! This chapter dives deep into state-of-the-art vision-language models and their applications in humanoid robotics. You'll learn how to create robots that can not only see and understand images but also comprehend and generate natural language descriptions, answer questions about visual scenes, and execute complex commands based on visual understanding.

## 🎯 Chapter Overview

### Learning Objectives
By the end of this chapter, you will be able to:
- Implement CLIP and contrastive learning for vision-language tasks
- Build Visual Question Answering systems for robots
- Create image captioning and scene understanding systems
- Apply grounded language understanding for robotics
- Develop multimodal reasoning capabilities
- Integrate vision-language models with robot control systems

### Prerequisites
- **Chapter 16**: Multimodal AI
- Deep learning and transformer architectures
- Computer vision fundamentals
- Natural language processing basics
- Python programming expertise

## 🔍 Vision-Language Model Foundations

### Contrastive Learning for Vision-Language

#### **CLIP Architecture**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class CLIPModel(nn.Module):
    def __init__(self, vision_encoder_name='openai/clip-vit-base-patch32',
                 text_encoder_name='openai/clip-vit-base-patch32',
                 embedding_dim=512):
        super().__init__()

        # Load pre-trained encoders
        self.vision_encoder = AutoModel.from_pretrained(vision_encoder_name)
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)

        # Get embedding dimensions
        vision_dim = self.vision_encoder.config.hidden_size
        text_dim = self.text_encoder.config.hidden_size

        # Projection layers to common embedding space
        self.vision_projection = nn.Linear(vision_dim, embedding_dim)
        self.text_projection = nn.Linear(text_dim, embedding_dim)

        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Robot-specific adaptations
        self.robot_adaptation = RobotVisionLanguageAdapter(embedding_dim)

    def encode_image(self, images):
        """Encode images to embedding space"""
        vision_outputs = self.vision_encoder(pixel_values=images)
        # Use [CLS] token or pooled output
        vision_features = vision_outputs.pooler_output
        vision_embeddings = self.vision_projection(vision_features)
        return F.normalize(vision_embeddings, dim=-1)

    def encode_text(self, texts):
        """Encode text to embedding space"""
        text_outputs = self.text_encoder(**texts)
        # Use [CLS] token or pooled output
        text_features = text_outputs.pooler_output
        text_embeddings = self.text_projection(text_features)
        return F.normalize(text_embeddings, dim=-1)

    def forward(self, images, texts):
        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(texts)

        # Calculate similarity matrix
        logit_scale = self.logit_scale.exp()
        similarity_matrix = torch.matmul(
            image_embeddings, text_embeddings.t()
        ) * logit_scale

        # Apply robot adaptation for action understanding
        adapted_features = self.robot_adaptation(image_embeddings, text_embeddings)

        return {
            'similarity_matrix': similarity_matrix,
            'image_embeddings': image_embeddings,
            'text_embeddings': text_embeddings,
            'adapted_features': adapted_features
        }

class RobotVisionLanguageAdapter(nn.Module):
    def __init__(self, embedding_dim, num_actions=10):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Cross-modal attention for robot understanding
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=8, batch_first=True
        )

        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, num_actions)
        )

        # Scene understanding head
        self.scene_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 50)  # Scene categories
        )

    def forward(self, image_embeddings, text_embeddings):
        # Reshape for attention
        image_expanded = image_embeddings.unsqueeze(1)  # [B, 1, D]
        text_expanded = text_embeddings.unsqueeze(1)    # [B, 1, D]

        # Cross-attention between vision and language
        attended_features, _ = self.cross_attention(
            image_expanded, text_expanded, text_expanded
        )

        # Concatenate original and attended features
        combined_features = torch.cat([
            image_expanded.squeeze(1), attended_features.squeeze(1)
        ], dim=-1)

        # Predict actions and scene understanding
        actions = self.action_head(combined_features)
        scene_understanding = self.scene_head(combined_features)

        return {
            'actions': actions,
            'scene_understanding': scene_understanding,
            'combined_features': combined_features
        }
```

#### **Training CLIP for Robotics**

```python
class CLIPTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )

        # Loss function
        self.contrastive_loss = ContrastiveLoss()

    def contrastive_loss(self, similarity_matrix):
        """Calculate contrastive loss for CLIP training"""
        batch_size = similarity_matrix.size(0)

        # Labels are diagonal (image i matches text i)
        labels = torch.arange(batch_size, device=similarity_matrix.device)

        # Calculate loss in both directions
        loss_i2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2i = F.cross_entropy(similarity_matrix.t(), labels)

        return (loss_i2t + loss_t2i) / 2

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['images'].to(self.device)
            texts = batch['texts'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images, texts)
            similarity_matrix = outputs['similarity_matrix']

            # Calculate contrastive loss
            loss = self.contrastive_loss(similarity_matrix)

            # Add robot-specific losses if available
            if 'robot_labels' in batch:
                robot_loss = self.calculate_robot_loss(
                    outputs['adapted_features'], batch['robot_labels']
                )
                loss += 0.1 * robot_loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Log progress
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')

        return total_loss / num_batches

    def calculate_robot_loss(self, features, labels):
        """Calculate robot-specific losses"""
        loss = 0

        if 'actions' in labels:
            action_loss = F.cross_entropy(
                features['actions'], labels['actions']
            )
            loss += action_loss

        if 'scenes' in labels:
            scene_loss = F.cross_entropy(
                features['scene_understanding'], labels['scenes']
            )
            loss += scene_loss

        return loss

    def evaluate(self):
        """Evaluate model performance"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['images'].to(self.device)
                texts = batch['texts'].to(self.device)

                outputs = self.model(images, texts)
                similarity_matrix = outputs['similarity_matrix']

                # Calculate loss
                loss = self.contrastive_loss(similarity_matrix)
                total_loss += loss.item()

                # Calculate accuracy (image-text matching)
                predictions = torch.argmax(similarity_matrix, dim=1)
                labels = torch.arange(images.size(0), device=self.device)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += images.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_predictions

        return avg_loss, accuracy
```

## ❓ Visual Question Answering (VQA)

### VQA Architecture

#### **Multimodal VQA Model**

```python
class VQAModel(nn.Module):
    def __init__(self, vision_encoder_name, text_encoder_name, vocab_size,
                 max_answer_length=20, hidden_dim=768):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_answer_length = max_answer_length
        self.hidden_dim = hidden_dim

        # Vision encoder
        self.vision_encoder = AutoModel.from_pretrained(vision_encoder_name)
        vision_dim = self.vision_encoder.config.hidden_size

        # Text encoder for questions
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        text_dim = self.text_encoder.config.hidden_size

        # Project to common dimension
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=12, batch_first=True
        )

        # Answer decoder
        self.answer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=12,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6
        )

        # Output vocabulary projection
        self.vocab_projection = nn.Linear(hidden_dim, vocab_size)

        # Robot-specific components
        self.robot_action_predictor = RobotActionPredictor(hidden_dim)

    def forward(self, images, questions, answers=None, max_length=None):
        batch_size = images.size(0)
        device = images.device

        # Encode images
        vision_outputs = self.vision_encoder(pixel_values=images)
        vision_features = vision_outputs.last_hidden_state  # [B, N_vision, D]
        vision_features = self.vision_proj(vision_features)

        # Encode questions
        question_outputs = self.text_encoder(**questions)
        question_features = question_outputs.last_hidden_state  # [B, N_question, D]
        question_features = self.text_proj(question_features)

        # Cross-modal attention: vision attends to question
        attended_vision, attention_weights = self.cross_attention(
            vision_features, question_features, question_features
        )

        # Use attended vision features as memory for decoder
        memory = attended_vision

        if answers is not None:
            # Training mode: teacher forcing
            answer_inputs = answers['input_ids']
            answer_features = self.text_encoder.embeddings(answer_inputs)

            # Decode answers
            decoded_features = self.answer_decoder(
                answer_features, memory
            )

            # Project to vocabulary
            logits = self.vocab_projection(decoded_features)

            return {
                'logits': logits,
                'attention_weights': attention_weights,
                'robot_actions': self.robot_action_predictor(memory.mean(dim=1))
            }

        else:
            # Inference mode: autoregressive generation
            return self.generate_answers(
                memory, max_length or self.max_answer_length, device
            )

    def generate_answers(self, memory, max_length, device):
        batch_size = memory.size(0)
        device = memory.device

        # Start with SOS token
        generated_ids = torch.zeros(
            batch_size, 1, dtype=torch.long, device=device
        )

        # Add SOS token (assuming token_id=0)
        generated_ids[:, 0] = 0

        for _ in range(max_length - 1):
            # Embed current generated tokens
            current_features = self.text_encoder.embeddings(generated_ids)

            # Decode next token
            decoded_features = self.answer_decoder(
                current_features, memory
            )

            # Get next token logits
            next_token_logits = self.vocab_projection(decoded_features[:, -1:, :])
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_tokens], dim=1)

            # Check for EOS token (assuming token_id=1)
            if (next_tokens == 1).all():
                break

        return {
            'generated_ids': generated_ids,
            'robot_actions': self.robot_action_predictor(memory.mean(dim=1))
        }

class RobotActionPredictor(nn.Module):
    def __init__(self, hidden_dim, num_actions=10):
        super().__init__()
        self.action_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_actions)
        )

        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        actions = self.action_predictor(features)
        confidence = self.confidence_estimator(features)
        return {'actions': actions, 'confidence': confidence}
```

#### **Robot VQA Integration**

```python
class RobotVQAController:
    def __init__(self, vqa_model, tokenizer, device='cuda'):
        self.model = vqa_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

        # Robot interface
        self.robot_interface = RobotInterface()

        # Question templates for robot interaction
        self.question_templates = {
            'object_recognition': [
                "What do you see in this image?",
                "What objects are present?",
                "Can you identify the main objects?"
            ],
            'spatial_reasoning': [
                "Where is the {object}?",
                "What is to the {direction} of the {object}?",
                "How far is the {object} from the {reference}?"
            ],
            'action_planning': [
                "What should I do with this {object}?",
                "How can I reach the {target}?",
                "What action would you take in this situation?"
            ],
            'safety_analysis': [
                "Is this situation safe?",
                "Are there any dangers present?",
                "What precautions should I take?"
            ]
        }

    def process_vision_input(self, image):
        """Process camera input for VQA"""
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        image_tensor = transform(image).unsqueeze(0).to(self.device)
        return image_tensor

    def process_question(self, question_text):
        """Process question text for VQA"""
        # Tokenize question
        inputs = self.tokenizer(
            question_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=77
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def answer_question(self, image, question_text):
        """Answer question about image"""
        # Process inputs
        image_tensor = self.process_vision_input(image)
        question_inputs = self.process_question(question_text)

        # Generate answer
        with torch.no_grad():
            outputs = self.model(image_tensor, question_inputs)

        # Decode answer
        answer_ids = outputs['generated_ids'][0]
        answer_text = self.tokenizer.decode(answer_ids, skip_special_tokens=True)

        # Extract robot actions if any
        robot_actions = outputs.get('robot_actions', {}).get('actions')
        confidence = outputs.get('robot_actions', {}).get('confidence')

        return {
            'answer': answer_text,
            'robot_actions': robot_actions,
            'confidence': confidence
        }

    def interactive_vqa_session(self):
        """Interactive VQA session with robot"""
        print("Starting Robot VQA Session. Type 'quit' to exit.")

        while True:
            # Get camera input
            image = self.capture_current_view()
            if image is None:
                print("Failed to capture image. Retrying...")
                continue

            # Show captured image (optional)
            plt.imshow(image)
            plt.title("Current View")
            plt.show()

            # Get question from user
            question = input("\nEnter your question: ").strip()

            if question.lower() == 'quit':
                break

            if not question:
                continue

            # Process question
            result = self.answer_question(image, question)

            print(f"\nAnswer: {result['answer']}")

            if result['robot_actions'] is not None:
                print(f"Suggested Action: {result['robot_actions']}")
                if result['confidence'] is not None:
                    print(f"Confidence: {result['confidence']:.2f}")

                # Ask if user wants to execute action
                execute = input("Execute this action? (y/n): ").lower()
                if execute == 'y':
                    self.execute_robot_action(result['robot_actions'])

    def capture_current_view(self):
        """Capture current robot camera view"""
        # Implement camera capture logic
        # This would integrate with the robot's camera system
        try:
            # Example using OpenCV (would be replaced with robot's camera API)
            import cv2
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()

            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(frame_rgb)
            else:
                return None
        except Exception as e:
            print(f"Camera capture error: {e}")
            return None

    def execute_robot_action(self, action_prediction):
        """Execute robot action based on VQA result"""
        try:
            # Convert action prediction to robot command
            action_index = torch.argmax(action_prediction).item()

            # Execute action through robot interface
            self.robot_interface.execute_action(action_index)
            print(f"Executed action {action_index}")
        except Exception as e:
            print(f"Failed to execute action: {e}")

    def scenario_based_qa(self, scenario_images, scenario_questions):
        """Process predefined scenarios for testing"""
        results = []

        for i, (image, questions) in enumerate(zip(scenario_images, scenario_questions)):
            print(f"\nProcessing Scenario {i+1}:")

            scenario_results = {
                'scenario_id': i+1,
                'image_path': image if isinstance(image, str) else 'captured',
                'questions': []
            }

            for j, question in enumerate(questions):
                print(f"  Q{j+1}: {question}")

                result = self.answer_question(image, question)

                print(f"  A{j+1}: {result['answer']}")

                scenario_results['questions'].append({
                    'question': question,
                    'answer': result['answer'],
                    'robot_actions': result['robot_actions'],
                    'confidence': result['confidence']
                })

            results.append(scenario_results)

        return results
```

## 🖼️ Image Captioning and Scene Understanding

### Image Captioning Models

#### **Encoder-Decoder Captioning**

```python
class ImageCaptioningModel(nn.Module):
    def __init__(self, vision_encoder_name, vocab_size, max_length=50,
                 hidden_dim=768, attention_dim=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim

        # Vision encoder
        self.vision_encoder = AutoModel.from_pretrained(vision_encoder_name)
        vision_dim = self.vision_encoder.config.hidden_size

        # Vision feature projection
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)

        # Attention mechanism
        self.attention = AttentionMechanism(hidden_dim, attention_dim)

        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=hidden_dim + hidden_dim,  # word embed + attention
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )

        # Word embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        # Scene understanding components
        self.scene_analyzer = SceneAnalyzer(hidden_dim)
        self.object_detector = ObjectDetector(hidden_dim)

    def forward(self, images, captions=None, teacher_forcing_ratio=0.5):
        batch_size = images.size(0)
        device = images.device

        # Extract visual features
        vision_outputs = self.vision_encoder(pixel_values=images)
        vision_features = vision_outputs.last_hidden_state  # [B, N_vision, D]
        vision_features = self.vision_proj(vision_features)

        # Scene analysis
        scene_info = self.scene_analyzer(vision_features)

        # Object detection
        object_info = self.object_detector(vision_features)

        if captions is not None:
            # Training mode
            caption_lengths = captions['attention_mask'].sum(dim=1)
            max_length = caption_lengths.max().item()

            # Initialize decoder hidden state
            hidden = self.init_hidden(batch_size, device)

            # Store outputs
            outputs = torch.zeros(
                batch_size, max_length, self.vocab_size, device=device
            )

            # Initialize input with SOS token
            input_token = torch.zeros(batch_size, dtype=torch.long, device=device)

            for t in range(max_length):
                # Get word embedding
                word_embed = self.embedding(input_token)

                # Apply attention
                context, attention_weights = self.attention(
                    hidden[0], vision_features
                )

                # Combine word embedding and context
                decoder_input = torch.cat([word_embed, context], dim=1)

                # Decoder step
                output, hidden = self.decoder(
                    decoder_input.unsqueeze(1), hidden
                )

                # Generate word probabilities
                word_probs = self.output_proj(output.squeeze(1))
                outputs[:, t, :] = word_probs

                # Teacher forcing
                use_teacher_forcing = (
                    torch.rand(1).item() < teacher_forcing_ratio
                )
                if use_teacher_forcing:
                    input_token = captions['input_ids'][:, t]
                else:
                    input_token = torch.argmax(word_probs, dim=1)

            return {
                'outputs': outputs,
                'attention_weights': attention_weights,
                'scene_info': scene_info,
                'object_info': object_info
            }

        else:
            # Inference mode
            return self.generate_caption(
                vision_features, scene_info, object_info, device
            )

    def init_hidden(self, batch_size, device):
        """Initialize hidden state for LSTM"""
        h = torch.zeros(2, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(2, batch_size, self.hidden_dim, device=device)
        return (h, c)

    def generate_caption(self, vision_features, scene_info, object_info, device):
        batch_size = vision_features.size(0)

        # Initialize
        hidden = self.init_hidden(batch_size, device)
        input_token = torch.zeros(batch_size, dtype=torch.long, device=device)

        generated_tokens = []
        attention_weights = []

        for _ in range(self.max_length):
            # Get word embedding
            word_embed = self.embedding(input_token)

            # Apply attention
            context, attn_weights = self.attention(hidden[0], vision_features)
            attention_weights.append(attn_weights)

            # Combine and decode
            decoder_input = torch.cat([word_embed, context], dim=1)
            output, hidden = self.decoder(decoder_input.unsqueeze(1), hidden)

            # Generate next token
            word_probs = self.output_proj(output.squeeze(1))
            next_token = torch.argmax(word_probs, dim=1)
            generated_tokens.append(next_token)

            # Stop if EOS token generated
            if (next_token == self.eos_token_id).all():
                break

            input_token = next_token

        generated_tokens = torch.stack(generated_tokens, dim=1)

        return {
            'generated_ids': generated_tokens,
            'attention_weights': torch.stack(attention_weights, dim=1),
            'scene_info': scene_info,
            'object_info': object_info
        }

class AttentionMechanism(nn.Module):
    def __init__(self, hidden_dim, attention_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim

        # Attention layers
        self.Wa = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.Ua = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.va = nn.Linear(attention_dim, 1)

    def forward(self, hidden, encoder_outputs):
        # hidden: [B, hidden_dim] (current decoder hidden state)
        # encoder_outputs: [B, seq_len, hidden_dim] (visual features)

        # Expand hidden to match sequence length
        hidden_expanded = hidden.unsqueeze(1).expand(-1, encoder_outputs.size(1), -1)

        # Calculate attention scores
        energy = torch.tanh(
            self.Wa(hidden_expanded) + self.Ua(encoder_outputs)
        )
        attention_scores = self.va(energy).squeeze(2)

        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=1)

        # Calculate context vector
        context = torch.bmm(
            attention_weights.unsqueeze(1), encoder_outputs
        ).squeeze(1)

        return context, attention_weights

class SceneAnalyzer(nn.Module):
    def __init__(self, hidden_dim, num_scene_classes=50):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.scene_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_scene_classes)
        )

    def forward(self, features):
        # Global average pooling
        pooled = self.global_pool(features.transpose(1, 2)).squeeze(-1)
        scene_logits = self.scene_classifier(pooled)
        return {'scene_logits': scene_logits, 'pooled_features': pooled}

class ObjectDetector(nn.Module):
    def __init__(self, hidden_dim, num_objects=80):
        super().__init__()
        self.object_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_objects)
        )

    def forward(self, features):
        # Classify each spatial location
        object_logits = self.object_classifier(features)
        return {'object_logits': object_logits}
```

### Robot Scene Understanding

```python
class RobotSceneUnderstanding:
    def __init__(self, captioning_model, tokenizer, device='cuda'):
        self.model = captioning_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

        # Action planning based on scene understanding
        self.action_planner = SceneBasedActionPlanner()

        # Scene memory for context
        self.scene_memory = SceneMemory()

    def understand_scene(self, image):
        """Generate comprehensive scene understanding"""
        # Preprocess image
        image_tensor = self.preprocess_image(image)

        with torch.no_grad():
            outputs = self.model.generate_caption(image_tensor)

        # Decode caption
        caption_ids = outputs['generated_ids'][0]
        caption = self.tokenizer.decode(caption_ids, skip_special_tokens=True)

        # Extract scene information
        scene_info = outputs['scene_info']
        object_info = outputs['object_info']

        # Generate action plan
        action_plan = self.action_planner.plan_actions(
            caption, scene_info, object_info
        )

        # Store in scene memory
        scene_id = self.scene_memory.store_scene(
            image, caption, scene_info, object_info, action_plan
        )

        return {
            'scene_id': scene_id,
            'caption': caption,
            'scene_info': scene_info,
            'object_info': object_info,
            'action_plan': action_plan,
            'attention_weights': outputs['attention_weights']
        }

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        image_tensor = transform(image).unsqueeze(0).to(self.device)
        return image_tensor

    def continuous_scene_monitoring(self):
        """Continuously monitor and understand scenes"""
        print("Starting continuous scene monitoring...")

        while True:
            try:
                # Capture current scene
                image = self.capture_scene()
                if image is None:
                    continue

                # Understand scene
                scene_understanding = self.understand_scene(image)

                # Print scene information
                print(f"\nScene ID: {scene_understanding['scene_id']}")
                print(f"Caption: {scene_understanding['caption']}")

                if scene_understanding['action_plan']['actions']:
                    print("Suggested Actions:")
                    for action in scene_understanding['action_plan']['actions']:
                        print(f"  - {action['description']} (confidence: {action['confidence']:.2f})")

                # Check for user input
                if input("Continue monitoring? (q to quit): ").lower() == 'q':
                    break

                time.sleep(2)  # Monitor every 2 seconds

            except KeyboardInterrupt:
                print("\nStopping scene monitoring...")
                break
            except Exception as e:
                print(f"Error in scene monitoring: {e}")
                continue

    def capture_scene(self):
        """Capture current scene from robot camera"""
        # Implement camera capture logic
        try:
            # This would integrate with robot's camera system
            import cv2
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(frame_rgb)
            else:
                return None
        except Exception as e:
            print(f"Camera capture error: {e}")
            return None

class SceneBasedActionPlanner:
    def __init__(self):
        # Action templates based on scene understanding
        self.action_templates = {
            'navigation': [
                'Navigate towards {object}',
                'Move to the {location}',
                'Approach the {object}'
            ],
            'manipulation': [
                'Pick up the {object}',
                'Move the {object} to {location}',
                'Grasp the {object}'
            ],
            'inspection': [
                'Examine the {object}',
                'Look at the {location}',
                'Inspect the {object} closely'
            ],
            'safety': [
                'Avoid the {object}',
                'Move away from danger',
                'Maintain safe distance'
            ]
        }

    def plan_actions(self, caption, scene_info, object_info):
        """Plan actions based on scene understanding"""
        actions = []

        # Extract key information from caption
        caption_lower = caption.lower()

        # Navigation actions
        if any(word in caption_lower for word in ['door', 'table', 'chair', 'path']):
            objects = self.extract_objects(caption)
            for obj in objects:
                if obj in ['door', 'table', 'chair']:
                    actions.append({
                        'type': 'navigation',
                        'description': f'Navigate towards {obj}',
                        'confidence': 0.8
                    })

        # Manipulation actions
        if any(word in caption_lower for word in ['pick', 'grab', 'move', 'take']):
            objects = self.extract_objects(caption)
            for obj in objects:
                actions.append({
                    'type': 'manipulation',
                    'description': f'Pick up {obj}',
                    'confidence': 0.7
                })

        # Inspection actions
        if any(word in caption_lower for word in ['look', 'see', 'examine', 'inspect']):
            actions.append({
                'type': 'inspection',
                'description': 'Examine the scene',
                'confidence': 0.9
            })

        # Safety actions based on scene analysis
        if self.detect_safety_concerns(scene_info, object_info):
            actions.append({
                'type': 'safety',
                'description': 'Maintain safe distance',
                'confidence': 0.95
            })

        return {
            'actions': actions,
            'scene_type': self.classify_scene_type(scene_info),
            'detected_objects': self.extract_dominant_objects(object_info)
        }

    def extract_objects(self, caption):
        """Extract objects from caption text"""
        # Simple keyword-based object extraction
        # In practice, this would use NLP techniques
        objects = []
        keywords = ['door', 'table', 'chair', 'person', 'window', 'floor', 'wall', 'object']

        caption_words = caption.lower().split()
        for word in caption_words:
            if word in keywords and word not in objects:
                objects.append(word)

        return objects

    def detect_safety_concerns(self, scene_info, object_info):
        """Detect potential safety concerns in the scene"""
        # Simple heuristic-based safety detection
        # In practice, this would use trained safety classifiers
        return False  # Placeholder

    def classify_scene_type(self, scene_info):
        """Classify the type of scene"""
        # Get most likely scene class
        if 'scene_logits' in scene_info:
            scene_logits = scene_info['scene_logits']
            scene_class = torch.argmax(scene_logits).item()
            return scene_class
        return 'unknown'

    def extract_dominant_objects(self, object_info):
        """Extract dominant objects from object detection"""
        # Get top detected objects
        if 'object_logits' in object_info:
            object_logits = object_info['object_logits']
            top_objects = torch.topk(object_logits.mean(dim=1), k=3)
            return top_objects.indices.tolist()
        return []

class SceneMemory:
    def __init__(self, max_scenes=100):
        self.scenes = {}
        self.max_scenes = max_scenes
        self.scene_counter = 0

    def store_scene(self, image, caption, scene_info, object_info, action_plan):
        """Store scene information in memory"""
        scene_id = self.scene_counter

        self.scenes[scene_id] = {
            'timestamp': time.time(),
            'image': image,  # Store image reference or features
            'caption': caption,
            'scene_info': scene_info,
            'object_info': object_info,
            'action_plan': action_plan
        }

        # Remove oldest scenes if exceeding max
        if len(self.scenes) > self.max_scenes:
            oldest_id = min(self.scenes.keys())
            del self.scenes[oldest_id]

        self.scene_counter += 1
        return scene_id

    def retrieve_scene(self, scene_id):
        """Retrieve stored scene information"""
        return self.scenes.get(scene_id, None)

    def search_similar_scenes(self, query_caption, top_k=5):
        """Search for similar scenes in memory"""
        # Implement similarity search based on captions
        # This would use text similarity embeddings
        similar_scenes = []
        for scene_id, scene_data in self.scenes.items():
            similarity = self.calculate_caption_similarity(
                query_caption, scene_data['caption']
            )
            similar_scenes.append((scene_id, similarity))

        # Sort by similarity and return top_k
        similar_scenes.sort(key=lambda x: x[1], reverse=True)
        return similar_scenes[:top_k]

    def calculate_caption_similarity(self, caption1, caption2):
        """Calculate similarity between two captions"""
        # Simple overlap-based similarity
        # In practice, this would use semantic embeddings
        words1 = set(caption1.lower().split())
        words2 = set(caption2.lower().split())

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0
```

## 📊 Grounded Language Understanding

### Grounded Language Processing

```python
class GroundedLanguageModel(nn.Module):
    def __init__(self, vision_encoder, text_encoder, grounding_dim=512):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        # Get dimensions
        vision_dim = vision_encoder.config.hidden_size
        text_dim = text_encoder.config.hidden_size

        # Grounding projections
        self.vision_grounding = nn.Linear(vision_dim, grounding_dim)
        self.text_grounding = nn.Linear(text_dim, grounding_dim)

        # Cross-modal grounding
        self.grounding_attention = nn.MultiheadAttention(
            embed_dim=grounding_dim, num_heads=8, batch_first=True
        )

        # Grounded action prediction
        self.action_head = nn.Sequential(
            nn.Linear(grounding_dim * 2, grounding_dim),
            nn.ReLU(),
            nn.Linear(grounding_dim, 20)  # Number of possible actions
        )

    def forward(self, images, text_commands):
        # Encode modalities
        vision_features = self.vision_encoder(pixel_values=images).last_hidden_state
        text_features = self.text_encoder(**text_commands).last_hidden_state

        # Project to grounding space
        vision_grounding = self.vision_grounding(vision_features)
        text_grounding = self.text_grounding(text_features)

        # Cross-modal attention for grounding
        grounded_vision, _ = self.grounding_attention(
            vision_grounding, text_grounding, text_grounding
        )

        # Pool features
        vision_pooled = grounded_vision.mean(dim=1)
        text_pooled = text_grounding.mean(dim=1)

        # Combine and predict actions
        combined = torch.cat([vision_pooled, text_pooled], dim=-1)
        actions = self.action_head(combined)

        return {
            'actions': actions,
            'grounded_vision': grounded_vision,
            'vision_pooled': vision_pooled,
            'text_pooled': text_pooled
        }

class GroundedRobotController:
    def __init__(self, grounded_model, tokenizer, device='cuda'):
        self.model = grounded_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

        # Action executor
        self.action_executor = RobotActionExecutor()

        # Grounding visualization
        self.visualizer = GroundingVisualizer()

    def execute_grounded_command(self, image, command_text):
        """Execute robot command grounded in visual context"""
        # Process inputs
        image_tensor = self.preprocess_image(image)
        text_inputs = self.tokenizer(
            command_text,
            return_tensors='pt',
            padding=True,
            truncation=True
        )

        # Move to device
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor, text_inputs)

        # Execute predicted action
        action_logits = outputs['actions']
        predicted_action = torch.argmax(action_logits, dim=-1).item()

        # Execute action
        success = self.action_executor.execute_action(predicted_action)

        # Visualize grounding
        self.visualizer.visualize_grounding(
            image, outputs['grounded_vision'], command_text
        )

        return {
            'command': command_text,
            'predicted_action': predicted_action,
            'success': success,
            'grounding_heatmap': outputs['grounded_vision']
        }

    def preprocess_image(self, image):
        """Preprocess image for model"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return transform(image).unsqueeze(0)

class RobotActionExecutor:
    def __init__(self):
        # Map action indices to robot commands
        self.action_map = {
            0: 'move_forward',
            1: 'move_backward',
            2: 'turn_left',
            3: 'turn_right',
            4: 'pick_up',
            5: 'put_down',
            6: 'wave_hand',
            7: 'nod_head',
            8: 'point_at',
            9: 'grasp_object',
            10: 'release_object',
            11: 'look_up',
            12: 'look_down',
            13: 'step_left',
            14: 'step_right',
            15: 'raise_arm',
            16: 'lower_arm',
            17: 'rotate_wrist',
            18: 'open_gripper',
            19: 'close_gripper'
        }

    def execute_action(self, action_index):
        """Execute robot action based on index"""
        if action_index in self.action_map:
            action_name = self.action_map[action_index]

            # Send command to robot
            try:
                # This would integrate with actual robot control system
                print(f"Executing action: {action_name}")

                # Simulate action execution
                success = self.simulate_action_execution(action_name)

                return success
            except Exception as e:
                print(f"Failed to execute action {action_name}: {e}")
                return False
        else:
            print(f"Unknown action index: {action_index}")
            return False

    def simulate_action_execution(self, action_name):
        """Simulate action execution for testing"""
        # In practice, this would send actual commands to robot
        time.sleep(0.5)  # Simulate action duration
        return True  # Simulate successful execution

class GroundingVisualizer:
    def __init__(self):
        pass

    def visualize_grounding(self, image, grounding_features, command_text):
        """Visualize attention/grounding on image"""
        if isinstance(image, str):
            image = Image.open(image)

        # Convert to numpy array
        img_array = np.array(image)

        # Average pooling to get 2D attention map
        if len(grounding_features.shape) == 3:
            attention_map = grounding_features.mean(dim=-1).squeeze(0)
        else:
            attention_map = grounding_features.squeeze(0)

        # Resize attention map to image size
        attention_resized = cv2.resize(
            attention_map.cpu().numpy(),
            (img_array.shape[1], img_array.shape[0])
        )

        # Normalize attention map
        attention_normalized = (attention_resized - attention_resized.min()) / \
                            (attention_resized.max() - attention_resized.min())

        # Create heatmap
        heatmap = cv2.applyColorMap(
            (attention_normalized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        # Overlay on original image
        overlay = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)

        # Display
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(img_array)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title(f'Grounding Attention: "{command_text}"')
        plt.axis('off')

        plt.tight_layout()
        plt.show()
```

## 📋 Chapter Summary

### Key Takeaways

1. **CLIP and Contrastive Learning**
   - Vision-text alignment through contrastive learning
   - Zero-shot transfer capabilities
   - Robot action prediction from vision-language pairs

2. **Visual Question Answering**
   - Multimodal attention mechanisms
   - Autoregressive answer generation
   - Robot action suggestion from VQA results

3. **Image Captioning and Scene Understanding**
   - Encoder-decoder architectures with attention
   - Scene analysis and object detection
   - Action planning based on visual understanding

4. **Grounded Language Understanding**
   - Cross-modal grounding of language in visual context
   - Action execution from natural language commands
   - Attention visualization for interpretability

### Practical Applications

1. **Natural Robot Control**: Execute commands through natural language
2. **Scene Description**: Generate human-readable scene descriptions
3. **Visual Reasoning**: Answer questions about visual scenes
4. **Task Planning**: Plan actions based on visual understanding

### Next Steps

With vision-language models mastered, you're ready for Chapter 18: **Human-Robot Interaction**, where we'll explore social robotics, gesture recognition, and natural human-robot communication.

---

**Ready to proceed?** Continue with [Chapter 18: Human-Robot Interaction](18-human-robot-interaction.md) to create truly interactive humanoid robots! 🤖👥

**Pro Tip**: Vision-language models represent a fundamental breakthrough in AI, enabling robots to understand and communicate about visual scenes in natural ways. These capabilities are essential for creating robots that can truly interact with humans in intuitive and meaningful ways! ✨🎯