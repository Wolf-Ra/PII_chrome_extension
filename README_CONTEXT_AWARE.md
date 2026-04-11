# Context-Aware PII Redaction Chrome Extension

This enhanced Chrome extension automatically identifies website context using LLM analysis and applies intelligent PII redaction based on the specific website type and industry.

## 🚀 New Features

### 1. **Intelligent Website Context Analysis**
- **LLM-Powered Domain Analysis**: Uses Groq LLM to analyze website domains and extract detailed context
- **Website Type Classification**: Automatically categorizes websites (healthcare, financial, education, etc.)
- **Industry-Specific PII Detection**: Prioritizes relevant PII types based on website industry
- **Sensitivity Level Assessment**: Determines appropriate redaction strictness per website type

### 2. **Context-Aware PII Redaction**
- **Healthcare Sites**: Focuses on medical records, patient IDs, insurance information
- **Financial Sites**: Prioritizes account numbers, credit cards, transaction history
- **Educational Sites**: Targets academic records, student IDs, grades
- **Corporate Sites**: Identifies trade secrets, internal codes, confidential business data

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Chrome        │    │   Local Server   │    │   LLM Service   │
│   Extension     │───▶│   (FastAPI)      │───▶│   (Groq)        │
│                 │    │                  │    │                 │
│ • PDF Interceptor│   │ • Domain Analysis│   │ • Website       │
│ • Modal UI      │    │ • Context        │   │   Classification│
│ • File Upload   │    │   Processing     │   │ • PII Rules     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   PII Pipeline    │
                    │                  │
                    │ • OCR Processing │
                    │ • Context-Aware  │
                    │   Detection      │
                    │ • Smart Redaction│
                    └──────────────────┘
```

## 📋 Website Context Categories

### Healthcare
- **Examples**: mayoclinic.com, hospital.org, mychart.com
- **Primary PII**: medical_records, patient_id, insurance_info, ssn, medical_history
- **Sensitivity**: Critical
- **Special Rules**: Patient names/DOB redacted unless medically required

### Financial
- **Examples**: hdfcbank.com, chase.com, paypal.com
- **Primary PII**: account_numbers, credit_card, transaction_history, ssn
- **Sensitivity**: Critical
- **Special Rules**: Account holder names redacted unless transaction-critical

### Educational
- **Examples**: harvard.edu, coursera.org, studentportal.edu
- **Primary PII**: academic_records, student_id, grades, ssn
- **Sensitivity**: High
- **Special Rules**: Student names redacted unless required for academic context

### Corporate
- **Examples**: github.com, linkedin.com, company.com
- **Primary PII**: email, full_name, api_keys, corporate_confidential
- **Sensitivity**: Medium
- **Special Rules**: Company names retained if contextually relevant

## 🛠️ Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up LLM API Key
```bash
# For Windows
set GROQ_API_KEY=your-groq-api-key-here

# For Linux/Mac
export GROQ_API_KEY=your-groq-api-key-here
```

### 3. Start the Local Server
```bash
python server.py
```
Server will run on `http://127.0.0.1:8000`

### 4. Install Chrome Extension
1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select the `chrome_extension` folder

## 🎯 How It Works

### Step 1: Domain Analysis
When you upload a PDF to a website, the extension:
1. Captures the domain name
2. Sends it to the local server
3. Server analyzes domain using LLM
4. Returns structured website context

### Step 2: Context-Aware Processing
The pipeline uses the website context to:
1. **Prioritize PII Types**: Focus on industry-specific sensitive data
2. **Adjust Sensitivity**: Apply appropriate redaction strictness
3. **Apply Special Rules**: Handle industry-specific exceptions
4. **Generate Smart Prompts**: Tailor LLM prompts with context

### Step 3: Intelligent Redaction
The system:
1. Extracts text using OCR
2. Applies context-aware detection rules
3. Uses LLM with website-specific instructions
4. Generates redacted PDF with industry-appropriate filtering

## 📊 Example Scenarios

### Scenario 1: Healthcare Portal Upload
**Domain**: `mayoclinic.com`
**Context**: Healthcare, Hospital, Critical Sensitivity
**PII Focus**: Medical records, patient IDs, insurance info
**Result**: Patient names/DOB redacted, hospital names retained if medically relevant

### Scenario 2: Banking Application
**Domain**: `hdfcbank.com`
**Context**: Financial, Banking, Critical Sensitivity
**PII Focus**: Account numbers, credit cards, SSN
**Result**: Account numbers redacted, bank names retained for context

### Scenario 3: Educational Platform
**Domain**: `harvard.edu`
**Context**: Education, University, High Sensitivity
**PII Focus**: Student IDs, grades, academic records
**Result**: Student names redacted unless required for academic records

## 🧪 Testing

### Test Website Context Analysis
```bash
python test_website_context.py
```

### Test with Different Domains
The system automatically handles:
- `hdfcbank.com` → Financial banking context
- `mayoclinic.com` → Healthcare hospital context
- `github.com` → Corporate technology context
- `harvard.edu` → Educational university context

## 🔧 Configuration

### Website Context Customization
You can modify the `_get_website_context()` function in `server.py` to:
- Add new website types
- Customize PII type mappings
- Adjust sensitivity levels
- Modify LLM prompts

### Detection Rules
Update the detector prompts in `detector.py` to:
- Add industry-specific instructions
- Modify PII category definitions
- Adjust redaction criteria

## 📝 API Endpoints

### POST /redact
**Request**: Multipart form with:
- `file`: PDF file to process
- `domain`: Target website domain

**Response**: Redacted PDF file

**Process**:
1. Analyze domain for website context
2. Apply context-aware PII detection
3. Generate redacted PDF
4. Return processed file

## 🚨 Security Considerations

- **Local Processing**: All processing happens on your local machine
- **No Data Leakage**: Only domain names are sent to LLM, not document content
- **API Key Security**: Store Groq API key securely as environment variable
- **Temporary Files**: All temporary files are automatically cleaned up

## 🐛 Troubleshooting

### Common Issues

1. **"GROQ_API_KEY not found"**
   - Set the environment variable as shown in setup

2. **"Server connection failed"**
   - Ensure the local server is running on port 8000
   - Check firewall settings

3. **"Extension not working"**
   - Verify developer mode is enabled
   - Check extension permissions
   - Ensure server is running

### Debug Mode
Enable detailed logging by checking `pii_debug.log` for:
- LLM API responses
- Domain analysis results
- PII detection decisions

## 🔄 Updates and Maintenance

### Adding New Website Types
1. Update the LLM prompt in `_get_website_context()`
2. Add detection rules in `detector.py`
3. Test with sample domains

### Improving PII Detection
1. Review detection logs
2. Adjust category definitions
3. Fine-tune LLM prompts
4. Test with industry-specific documents

## 📄 License

This project maintains the same license as the original PII redaction system.

## 🤝 Contributing

Contributions welcome! Focus on:
- New website type support
- Improved detection accuracy
- Better user experience
- Enhanced security features
