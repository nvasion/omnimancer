# Omnimancer CLI Security Guide

This document outlines the security measures implemented in Omnimancer CLI and provides recommendations for secure usage.

## Security Features Implemented

### 1. API Key Protection

#### Masking and Sanitization
- API keys are automatically masked in logs and error messages
- Full API keys are never displayed in console output
- Error messages sanitize sensitive data using regex patterns
- Configuration display shows only masked versions of credentials

#### Format Validation
- OpenAI keys: Must start with `sk-` and be at least 40 characters
- Claude keys: Must start with `sk-ant-` and be at least 40 characters  
- AWS Access Keys: Must start with `AKIA` and be exactly 20 characters
- Invalid format keys are rejected before making API calls

### 2. Configuration Security

#### File Permissions
- Configuration files should use restricted permissions (600 or 640)
- Backup files inherit the same security permissions
- World-writable permissions are prevented

#### Serialization Safety
- Sensitive data masking functions available for display purposes
- Configuration can be serialized with or without sensitive data
- Environment variable support for credential storage

### 3. Provider-Specific Security

#### OpenAI
- API key format validation (`sk-*`)
- HTTPS-only communication
- Proper error handling without credential exposure

#### Claude (Anthropic)
- API key format validation (`sk-ant-*`)
- Secure endpoint communication
- Credential masking in all outputs

#### Azure OpenAI
- HTTPS endpoint validation
- API version specification required
- Azure-specific credential handling

#### AWS Bedrock
- Access key format validation (`AKIA*`)
- Secret key length validation
- Region specification required
- Credential masking for both access and secret keys

#### Google Vertex AI
- Service account JSON file validation
- Project and location specification
- Secure credential file handling

#### Other Providers
- Consistent HTTPS endpoint usage
- Provider-specific credential validation
- Uniform error handling and masking

### 4. Network Security

#### HTTPS Enforcement
All provider endpoints use HTTPS:
- OpenAI: `https://api.openai.com`
- Claude: `https://api.anthropic.com`
- Gemini: `https://generativelanguage.googleapis.com`
- Perplexity: `https://api.perplexity.ai`
- xAI: `https://api.x.ai`
- Mistral: `https://api.mistral.ai`
- OpenRouter: `https://openrouter.ai/api`

#### Request Headers
- No sensitive data included in standard headers
- Proper User-Agent identification
- Content-Type and Accept headers properly set

### 5. Error Handling Security

#### Sanitization Patterns
- OpenAI keys: `sk-[a-zA-Z0-9]{40,}` → `sk-***`
- Claude keys: `sk-ant-[a-zA-Z0-9-]{40,}` → `sk-ant-***`
- AWS keys: `AKIA[A-Z0-9]{16}` → `AKIA***`
- Long secrets: `[a-zA-Z0-9]{32,}` → `***`

#### Safe Error Messages
- Authentication errors show masked credentials only
- Network errors don't expose endpoint credentials
- Configuration errors mask sensitive values

## Security Recommendations

### 1. Credential Management

#### Environment Variables (Recommended)
```bash
export OPENAI_API_KEY="your-openai-key"
export CLAUDE_API_KEY="your-claude-key"
export AZURE_OPENAI_KEY="your-azure-key"
export AWS_ACCESS_KEY_ID="your-aws-access-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
```

#### Configuration File Security
```bash
# Set restrictive permissions
chmod 600 ~/.omnimancer/config.json

# Verify permissions
ls -la ~/.omnimancer/config.json
# Should show: -rw------- (600)
```

### 2. Production Deployment

#### Secure Storage Systems
- Use HashiCorp Vault for credential storage
- Implement AWS Secrets Manager integration
- Use Azure Key Vault for Azure deployments
- Consider Google Secret Manager for GCP

#### Key Rotation
- Rotate API keys regularly (monthly recommended)
- Use different keys for different environments
- Implement automated key rotation where possible
- Monitor key usage and access patterns

### 3. Development Best Practices

#### Local Development
- Never commit API keys to version control
- Use `.env` files with `.gitignore` entries
- Use separate development API keys
- Implement pre-commit hooks to scan for secrets

#### Testing
- Use mock providers for unit tests
- Implement integration tests with test keys
- Validate security measures in CI/CD pipelines
- Regular security audits and penetration testing

### 4. Monitoring and Auditing

#### Logging Security
- Never log full API keys or credentials
- Implement structured logging with sanitization
- Monitor for credential exposure in logs
- Set up alerts for authentication failures

#### Access Monitoring
- Track API key usage patterns
- Monitor for unusual access patterns
- Implement rate limiting to prevent abuse
- Log configuration changes and access

## Security Checklist

### Before Deployment
- [ ] All API keys stored securely (environment variables or vault)
- [ ] Configuration files have restrictive permissions
- [ ] No credentials committed to version control
- [ ] HTTPS endpoints verified for all providers
- [ ] Error handling tested for credential exposure
- [ ] Logging reviewed for sensitive data leaks

### Regular Maintenance
- [ ] API keys rotated according to schedule
- [ ] Security patches applied promptly
- [ ] Access logs reviewed for anomalies
- [ ] Configuration backups secured properly
- [ ] Dependencies updated for security fixes

### Incident Response
- [ ] Procedure for compromised credentials
- [ ] Key revocation and rotation process
- [ ] Incident logging and reporting
- [ ] Security team contact information
- [ ] Recovery and remediation steps

## Compliance Considerations

### Data Protection
- API keys and credentials are considered sensitive data
- Implement data retention policies for logs
- Consider GDPR/CCPA requirements for user data
- Encrypt sensitive data at rest and in transit

### Industry Standards
- Follow OWASP security guidelines
- Implement SOC 2 Type II controls where applicable
- Consider ISO 27001 compliance requirements
- Adhere to provider-specific security requirements

## Security Contact

For security-related issues or questions:
- Review this documentation first
- Check provider-specific security guidelines
- Implement recommended security measures
- Consider professional security consultation for production deployments

## Version History

- v1.0: Initial security implementation
- v1.1: Added provider-specific security measures
- v1.2: Enhanced error handling and sanitization
- v1.3: Added comprehensive security testing

---

**Note**: This security guide should be reviewed and updated regularly as new providers are added and security requirements evolve.