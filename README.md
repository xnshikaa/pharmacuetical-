# Pharmaceutical Risk Reporter with Image Recognition

A comprehensive web application for generating pharmaceutical contamination risk assessment reports with AI-powered image analysis for tablet defect detection.

## 🚀 Features

### Core Functionality
- **Image Upload & Analysis**: Upload tablet images for automated defect detection
- **Defect Detection**: AI-powered analysis of cracks, discoloration, contamination, deformation, and coating defects
- **Report Generation**: Comprehensive regulatory reports with image analysis integration
- **Multi-Agency Support**: Support for USFDA, MHRA, EMA, and other regulatory agencies
- **Report History**: Track and review all generated reports
- **Report Revisions**: Edit and improve reports based on feedback

### Image Recognition Capabilities
- **Crack Detection**: Identifies fractures and surface damage
- **Discoloration Analysis**: Detects abnormal color variations
- **Contamination Detection**: Finds foreign particles and debris
- **Shape Analysis**: Identifies deformation and size irregularities
- **Coating Defect Detection**: Analyzes coating uniformity and damage

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Ollama (for AI report generation)
- OpenCV and other ML libraries

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd pharma_report_app_image_recognition
   ```

2. **Create virtual environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Ollama server**
   ```bash
   ollama serve
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   - Open browser and go to `http://localhost:8081`
   - Login with username: `admin`, password: `admin`

## Usage Guide

### 1. Image Upload
1. Navigate to the "Upload Image" section
2. Upload a clear image of the tablet
3. The system will automatically analyze the image for defects
4. Review the analysis results

### 2. Report Generation
1. Fill in basic information (tablet name, batch number, agency)
2. Provide contamination details
3. Answer recall-related questions
4. Generate comprehensive report

### 3. Report Management
- View all reports in the History section
- Revise reports based on feedback
- Track image analysis results

## Image Requirements

For best analysis results, ensure your images meet these criteria:

- **Resolution**: Minimum 800x600 pixels
- **Lighting**: Well-lit, clear images
- **Background**: White or neutral background
- **Focus**: Tablet should be clearly visible and centered
- **Coverage**: Include both sides if defects are visible
- **Format**: PNG, JPG, JPEG, GIF, or BMP (max 16MB)

## Model Training

### Training Your Own Model

1. **Prepare Training Data**
   ```
   training_data/
   ├── cracks/
   │   ├── crack1.jpg
   │   └── crack2.png
   ├── discoloration/
   │   ├── discolored1.jpg
   │   └── discolored2.png
   ├── contamination/
   │   └── contaminated1.jpg
   ├── deformation/
   │   └── deformed1.jpg
   └── coating_defects/
       └── coating_defect1.jpg
   ```

2. **Run Training**
   ```bash
   python train_model.py --data_dir training_data --output results.json --report report.txt
   ```

3. **Review Results**
   - Check accuracy metrics
   - Review false positives/negatives
   - Adjust detection parameters

### Improving Model Performance

- **Add More Training Data**: Include diverse images of each defect type
- **Balance Dataset**: Ensure equal representation of all defect types
- **Quality Images**: Use high-quality, well-lit images
- **Adjust Parameters**: Modify thresholds in `image_processor.py`

## Architecture

### Components

1. **Flask Web Application** (`app.py`)
   - Handles user interface and form processing
   - Manages file uploads and sessions
   - Integrates with Ollama for report generation

2. **Image Processing Module** (`image_processor.py`)
   - Defect detection algorithms
   - Image preprocessing
   - Analysis result generation

3. **Training Module** (`train_model.py`)
   - Model evaluation and training
   - Performance metrics calculation
   - Training report generation

### Defect Detection Algorithms

- **Crack Detection**: Edge detection with morphological operations
- **Discoloration**: Color variance analysis in HSV space
- **Contamination**: Threshold-based dark region detection
- **Deformation**: Shape analysis using contour properties
- **Coating Defects**: Gradient-based surface irregularity detection

## Supported Regulatory Agencies

- **USFDA** (United States Food and Drug Administration)
- **MHRA** (United Kingdom Medicines and Healthcare products Regulatory Agency)
- **EMA** (European Medicines Agency)
- **PMDA** (Japan Pharmaceuticals and Medical Devices Agency)
- **TGA** (Australia Therapeutic Goods Administration)
- **Health Canada**
- **SFDA** (Saudi Arabia Saudi Food and Drug Authority)
- **Indian FDA**

## Configuration

### Environment Variables
- `FLASK_SECRET_KEY`: Secret key for session management
- `OLLAMA_URL`: Ollama server URL (default: http://localhost:11434)
- `UPLOAD_FOLDER`: Directory for uploaded images (default: uploads/)

### Model Parameters
Adjust detection sensitivity in `image_processor.py`:
- Crack detection threshold
- Discoloration variance threshold
- Contamination size limits
- Deformation circularity threshold
- Coating defect gradient threshold

## Security Considerations

- Change default admin credentials in production
- Use secure secret keys
- Implement proper authentication
- Validate file uploads
- Use HTTPS in production

## Performance Optimization

- **Image Preprocessing**: Resize large images for faster processing
- **Caching**: Cache analysis results for repeated images
- **Database**: Use proper database instead of in-memory storage
- **Load Balancing**: Scale for multiple users

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   - Ensure Ollama is running: `ollama serve`
   - Check if llama3 model is available: `ollama list`

2. **Image Analysis Fails**
   - Verify image format is supported
   - Check image file size (max 16MB)
   - Ensure image is not corrupted

3. **Poor Detection Accuracy**
   - Use higher quality images
   - Ensure proper lighting
   - Train model with more data

### Debug Mode
Run with debug enabled:
```bash
export FLASK_ENV=development
python app.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for computer vision capabilities
- Flask for web framework
- Ollama for AI model integration
- Pharmaceutical industry experts for domain knowledge

## Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the documentation

---

**Note**: This system is designed for educational and research purposes. For production use in pharmaceutical environments, ensure compliance with relevant regulations and conduct thorough validation. 