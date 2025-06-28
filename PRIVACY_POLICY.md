# Privacy Policy - Adaptive Graph of Thoughts Desktop Extension

**Effective Date**: December 2024  
**Last Updated**: December 2024

## Overview

The Adaptive Graph of Thoughts Desktop Extension ("the Extension") is designed with privacy and security as core principles. This privacy policy explains how the Extension handles your data and protects your privacy.

## Data Processing Philosophy

### Local-First Architecture
- **All processing happens locally** on your device
- **No data is transmitted** to remote servers by the Extension itself
- **Your research data stays on your computer** at all times
- **No user accounts or registration** required

## Data We Do NOT Collect

The Extension does **NOT** collect, store, or transmit:
- ❌ Personal information or user accounts
- ❌ Research queries or scientific data
- ❌ Neo4j database contents
- ❌ Configuration settings or API keys
- ❌ Usage analytics or telemetry
- ❌ Error reports or crash data
- ❌ Any files or documents you analyze

## Data Usage

### What Happens to Your Data
1. **Scientific Queries**: Processed locally using your Neo4j database
2. **Graph Analysis**: Performed on your local Neo4j instance
3. **Configuration**: Stored locally in your Claude Desktop settings
4. **Logs**: Generated locally and stored on your device only

### External API Calls (Optional)
If you configure external API keys (PubMed, Google Scholar, OpenAI, Anthropic):
- **Direct calls** are made from your device to these services
- **We do not proxy** or intercept these calls
- **Their privacy policies apply** to data sent to these services
- **You control** which external services to use

## Third-Party Services

### Optional Integrations
The Extension can optionally integrate with:

**Research Databases**:
- **PubMed/NCBI**: For scientific literature search
- **Google Scholar**: For academic paper discovery
- **Exa Search**: For web-based research

**AI Services**:
- **OpenAI**: For enhanced reasoning capabilities
- **Anthropic**: For additional AI analysis

**Important**: 
- These integrations are **entirely optional**
- You must **explicitly configure** API keys to enable them
- Data sent to these services is governed by **their privacy policies**
- We recommend reviewing their policies before enabling integrations

### Local Infrastructure
**Neo4j Database**: All graph data is stored in your local Neo4j instance
- **Private to your device**
- **Under your complete control**
- **Not accessible** to the Extension developers

## Data Security

### Local Data Protection
- **Encrypted storage**: API keys stored securely using OS keychain/credential manager
- **No data transmission**: All processing happens locally
- **Secure defaults**: Conservative security settings by default
- **Input validation**: All user inputs are validated and sanitized

### Network Security
- **HTTPS only**: All external API calls use secure connections
- **No proxy servers**: Direct communication with configured services
- **Request timeout**: 30-second limits prevent hanging connections
- **Error isolation**: Network errors don't expose sensitive data

## Your Privacy Rights

### Data Control
You have complete control over:
- ✅ **All data processing** happens on your device
- ✅ **Configuration choices** including which APIs to use
- ✅ **Data retention** - delete anytime by removing the Extension
- ✅ **External service integration** - enable/disable at will

### Data Portability
- **Neo4j data**: Standard graph database format, easily exportable
- **Configuration**: Stored in standard Claude Desktop format
- **No vendor lock-in**: Switch to other tools anytime

## Children's Privacy

The Extension is designed for academic and professional use. We do not:
- Knowingly collect data from children under 13
- Target children as our user base
- Include content inappropriate for children

## Changes to Privacy Policy

### Notification Process
- **GitHub repository**: Updates will be posted to the project repository
- **Version tracking**: Changes tracked in git history
- **Documentation updates**: Privacy policy updates reflected in all documentation

### Your Consent
By using the Extension, you agree to this privacy policy. If you disagree with any part of this policy, please do not use the Extension.

## Contact Information

### Privacy Questions
For privacy-related questions or concerns:
- **GitHub Issues**: https://github.com/SaptaDey/Adaptive-Graph-of-Thoughts-MCP-server/issues
- **Email**: [Your Contact Email]
- **Repository**: https://github.com/SaptaDey/Adaptive-Graph-of-Thoughts-MCP-server

### Security Vulnerabilities
To report security vulnerabilities:
- **Security Policy**: See SECURITY.md in the repository
- **Responsible disclosure**: We follow responsible disclosure practices
- **Response time**: Security issues addressed within 30 days

## Legal Compliance

### Applicable Law
This privacy policy is governed by applicable data protection laws including:
- **GDPR** (if applicable to EU users)
- **CCPA** (if applicable to California users)  
- **Local data protection laws** in your jurisdiction

### Open Source Transparency
- **MIT License**: Extension code is open source and auditable
- **No hidden functionality**: All code is publicly available
- **Community oversight**: Open to security audits and contributions

---

## Summary

**Your privacy is paramount**. The Adaptive Graph of Thoughts Extension is designed to:
- ✅ Process all data locally on your device
- ✅ Give you complete control over your research data
- ✅ Protect your privacy through local-first architecture
- ✅ Enable optional external integrations under your control
- ✅ Maintain transparency through open source code

**We never see your data. Your research stays private.**