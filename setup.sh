#!/bin/bash
echo "Generating dataset..."
python src/generate_data.py
echo "Training model..."
python src/train_model.py
echo "Setup complete!"
