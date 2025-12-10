# Deployment and Maintenance Guide

This guide covers the deployment process and ongoing maintenance procedures for the Physical AI & Humanoid Robotics educational platform.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Deployment Requirements](#deployment-requirements)
3. [Automated Deployment (GitHub Actions)](#automated-deployment-github-actions)
4. [Manual Deployment](#manual-deployment)
5. [Environment Configuration](#environment-configuration)
6. [Performance Monitoring](#performance-monitoring)
7. [Content Updates](#content-updates)
8. [Troubleshooting](#troubleshooting)
9. [Maintenance Schedule](#maintenance-schedule)

---

## Overview

The Physical AI & Humanoid Robotics platform is built with Docusaurus 3.9.2 and designed for deployment on GitHub Pages. The deployment process includes:

- Static site generation
- Asset optimization
- SEO optimization
- Accessibility compliance
- Performance monitoring

### Deployment Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Development   │───▶│  GitHub Actions  │───▶│  GitHub Pages   │
│   (Local)       │    │   (CI/CD)        │    │  (Production)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
    npm run build          Automated Tests        Static Site
         │                       │                       │
    Local Testing     Link Validation &      SEO & Performance
         │                   Accessibility           Optimization
```

---

## Deployment Requirements

### System Requirements

- **Node.js**: 16.x, 18.x, or 20.x
- **npm**: 8.x or higher
- **Git**: 2.x or higher
- **GitHub Account**: For deployment to GitHub Pages

### Repository Structure

Ensure your repository follows this structure:

```
humanoid-robotics/
├── .github/
│   └── workflows/          # CI/CD workflows
├── docs/                   # Documentation content
├── src/
│   ├── css/               # Custom styles
│   └── components/        # React components
├── static/                # Static assets
├── docusaurus.config.js   # Docusaurus configuration
├── sidebars.js           # Navigation structure
├── package.json          # Dependencies and scripts
└── README.md             # Project documentation
```

---

## Automated Deployment (GitHub Actions)

The platform uses GitHub Actions for automated deployment. The deployment process is triggered by:

1. **Push to main branch**: Automatic deployment
2. **Pull requests**: Build validation
3. **Manual workflow dispatch**: On-demand deployment

### Workflow Files

- **`.github/workflows/deploy.yml`**: Main deployment workflow
- **`.github/workflows/ci.yml`**: Continuous integration testing
- **`.github/workflows/link-check.yml`**: Link validation

### Deployment Process

1. **Code Quality Checks**
   - ESLint validation
   - TypeScript checking (if applicable)
   - Build verification

2. **Content Validation**
   - Link checking
   - Frontmatter validation
   - Image reference verification

3. **Security & Performance**
   - Dependency audit
   - Accessibility testing
   - Performance analysis

4. **Site Generation**
   - Docusaurus build
   - Asset optimization
   - Sitemap generation

5. **Deployment**
   - GitHub Pages deployment
   - URL configuration
   - SSL certificate management

---

## Manual Deployment

For manual deployment or local development:

### Prerequisites

```bash
# Install dependencies
npm install

# Start development server
npm run start
```

### Build Process

```bash
# Build for production
npm run build

# Serve locally (for testing)
npm run serve
```

### Manual Deployment Steps

1. **Build the site**:
   ```bash
   npm run build
   ```

2. **Test the build**:
   ```bash
   npm run serve
   # Visit http://localhost:3000 to verify
   ```

3. **Deploy to GitHub Pages**:
   ```bash
   # Using GitHub CLI
   gh pages deploy build

   # Or manually copy to gh-pages branch
   git subtree push --prefix build origin gh-pages
   ```

---

## Environment Configuration

### Required Environment Variables

For Context7 MCP integration and advanced features:

```bash
# Context7 MCP Server
CONTEXT7_MCP_SERVER=http://localhost:3001

# Analytics (optional)
GOOGLE_ANALYTICS_ID=G-XXXXXXXXXX

# Search functionality (optional)
ALGOLIA_APP_ID=your_app_id
ALGOLIA_API_KEY=your_api_key
ALGOLIA_INDEX_NAME=your_index_name
```

### GitHub Secrets

Configure these in your repository settings:

1. **GitHub Pages**: Automatically configured by GitHub Actions
2. **LHCI Token**: For Lighthouse CI (optional)
3. **Analytics Tokens**: For performance tracking (optional)

---

## Performance Monitoring

### Built-in Monitoring

The platform includes several monitoring mechanisms:

1. **Lighthouse CI**: Automated performance testing
2. **Bundle Analysis**: Build size tracking
3. **Link Validation**: Content integrity checking

### Performance Metrics

Monitor these key metrics:

- **First Contentful Paint (FCP)**: < 1.8s
- **Largest Contentful Paint (LCP)**: < 2.5s
- **Cumulative Layout Shift (CLS)**: < 0.1
- **First Input Delay (FID)**: < 100ms

### Monitoring Tools

1. **GitHub Actions**: CI/CD pipeline results
2. **Lighthouse CI**: Performance scores
3. **Google Analytics**: User engagement
4. **GitHub Insights**: Repository traffic

---

## Content Updates

### Adding New Content

1. **Create new markdown files** in appropriate directories
2. **Update sidebars.js** if adding new sections
3. **Test locally** before committing
4. **Push to main** for automatic deployment

### Content Guidelines

- Use consistent frontmatter format
- Include proper metadata (title, description, etc.)
- Validate all links and images
- Test code examples

### Version Control Strategy

```bash
# Feature development
git checkout -b feature/new-content
# Make changes...
git commit -m "Add new chapter on advanced perception"
git push origin feature/new-content
# Create pull request
```

---

## Troubleshooting

### Common Issues

#### Build Failures

**Problem**: Build fails with missing dependencies
```bash
# Solution
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Problem**: Link validation errors
```bash
# Solution
# Check .github/link-check-config.json for ignored patterns
# Update broken links in content
```

#### Deployment Issues

**Problem**: GitHub Pages not updating
```bash
# Solution
# Check GitHub Actions logs
# Verify repository settings for GitHub Pages
# Ensure branch protection rules allow deployment
```

**Problem**: Context7 MCP integration not working
```bash
# Solution
# Verify CONTEXT7_MCP_SERVER environment variable
# Check MCP server status
# Review network connectivity
```

#### Performance Issues

**Problem**: Slow loading times
```bash
# Solution
# Optimize images (use WebP format)
# Reduce bundle size
# Enable lazy loading
# Review third-party dependencies
```

### Debug Mode

Enable debug logging:

```bash
# Debug Docusaurus build
DEBUG=docusaurus:* npm run build

# Debug webpack bundling
DEBUG=webpack:* npm run build
```

---

## Maintenance Schedule

### Daily

- **Automated link checking**: Validates all internal and external links
- **Build verification**: Ensures site builds successfully
- **Security scanning**: Checks for vulnerable dependencies

### Weekly

- **Performance monitoring**: Review Lighthouse CI results
- **Content audit**: Check for outdated information
- **Analytics review**: Monitor user engagement metrics

### Monthly

- **Dependency updates**: Update npm packages
- **Security updates**: Apply security patches
- **Performance optimization**: Review and improve loading times

### Quarterly

- **Content review**: Comprehensive content audit
- **Accessibility testing**: WCAG compliance verification
- **SEO optimization**: Review search engine performance
- **Backup verification**: Ensure content is properly backed up

### Annual

- **Platform upgrade**: Update Docusaurus and major dependencies
- **Architecture review**: Evaluate technology choices
- **Security audit**: Comprehensive security assessment
- **Performance baseline**: Establish new performance targets

---

## Emergency Procedures

### Site Down

1. **Check GitHub Actions status**
2. **Review recent commits for breaking changes**
3. **Rollback if necessary**:
   ```bash
   git revert <commit-hash>
   git push origin main
   ```

### Security Incident

1. **Immediate actions**:
   - Rotate all secrets and API keys
   - Review access logs
   - Update dependencies

2. **Post-incident**:
   - Document the incident
   - Implement preventive measures
   - Update security policies

### Content Recovery

1. **Restore from Git history**:
   ```bash
   git log --oneline --follow docs/chapter.md
   git checkout <commit-hash> -- docs/chapter.md
   ```

2. **Backup restoration**:
   - GitHub repository backup
   - Local development backup
   - Content management system backup (if used)

---

## Best Practices

### Development

1. **Feature branches**: Use separate branches for new features
2. **Pull requests**: Code review before merging
3. **Testing**: Validate locally before pushing
4. **Documentation**: Update docs with new features

### Content Management

1. **Version control**: Track all content changes
2. **Link validation**: Ensure all links work
3. **Image optimization**: Use appropriate formats and sizes
4. **Accessibility**: Follow WCAG 2.1 guidelines

### Performance

1. **Lazy loading**: Implement for images and components
2. **Code splitting**: Optimize bundle sizes
3. **Caching**: Implement proper caching strategies
4. **CDN**: Use content delivery networks when applicable

### Security

1. **Dependency scanning**: Regular security audits
2. **Access control**: Limit deployment permissions
3. **Secrets management**: Use environment variables for sensitive data
4. **Regular updates**: Keep dependencies up to date

---

## Support and Resources

### Documentation

- **Docusaurus Documentation**: https://docusaurus.io/docs
- **GitHub Actions Documentation**: https://docs.github.com/en/actions
- **MDN Web Docs**: https://developer.mozilla.org/

### Tools and Utilities

- **Lighthouse**: https://developers.google.com/web/tools/lighthouse
- **WebPageTest**: https://www.webpagetest.org/
- **GTmetrix**: https://gtmetrix.com/

### Community

- **Docusaurus Discord**: Community support and discussions
- **GitHub Issues**: Report bugs and request features
- **Stack Overflow**: Technical questions and answers

---

*This deployment guide is maintained as part of the Physical AI & Humanoid Robotics educational platform. For the most up-to-date information, please refer to the GitHub repository.*