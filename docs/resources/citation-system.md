---
title: "Citation System and Guidelines"
sidebar_label: "Citation System"
sidebar_position: 8
---

# Citation System and Guidelines

This document outlines the citation system used throughout the humanoid robotics educational guide, providing standardized formats for referencing academic work and enabling easy access to source materials through clickable links.

## 📑 Table of Contents

1. [Citation Format Standards](#1-citation-format-standards)
2. [In-Text Citation System](#2-in-text-citation-system)
3. [Reference List Guidelines](#3-reference-list-guidelines)
4. [Digital Object Identifiers (DOIs)](#4-digital-object-identifiers-dois)
5. [Online Resource Citation](#5-online-resource-citation)
6. [Software and Code Citation](#6-software-and-code-citation)
7. [Citation Management Tools](#7-citation-management-tools)

---

## 1. Citation Format Standards

### **APA 7th Edition Format**

This guide follows APA 7th edition formatting standards with modifications for digital accessibility:

#### **Journal Articles**
```
Author, A. A., & Author, B. B. (Year). Title of the article. *Journal Name, Volume*(Issue), pages. https://doi.org/xxxxx
```

**Example:**
```
Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. *International Journal of Computer Vision, 60*(2), 91-110. https://doi.org/10.1023/B:VISI.0000029664.99615.94
```

#### **Conference Papers**
```
Author, A. A., & Author, B. B. (Year, Month). *Title of the paper*. Paper presented at the Conference Name, City, State. https://doi.org/xxxxx
```

**Example:**
```
Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011, May). *ORB: An efficient alternative to SIFT or SURF*. Paper presented at the IEEE International Conference on Computer Vision, Barcelona, Spain. https://doi.org/10.1109/ICCV.2011.6126544
```

#### **Books**
```
Author, A. A. (Year). *Title of work* (Edition). Publisher. https://doi.org/xxxxx
```

**Example:**
```
Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic robotics*. MIT Press. https://doi.org/10.1109/MRA.2011.941932
```

#### **Technical Reports**
```
Author, A. A., & Author, B. B. (Year). *Title of report* (Report No. XXX). Publisher. URL
```

---

## 2. In-Text Citation System

### **Parenthetical Citations**
```markdown
Recent advances in computer vision have revolutionized robotic perception (Lowe, 2004).

Multiple studies have demonstrated the effectiveness of deep learning approaches (Krizhevsky et al., 2012; He et al., 2015).
```

### **Narrative Citations**
```markdown
Lowe (2004) developed the SIFT algorithm for feature detection.

Radford et al. (2021) demonstrated that large-scale pre-training could enable zero-shot capabilities in vision models.
```

### **Citation with Page Numbers**
```markdown
The probabilistic robotics framework provides a systematic approach to state estimation (Thrun et al., 2005, p. 45).

Sensor fusion techniques typically follow either centralized or decentralized architectures (Smith & Jones, 2020, pp. 112-115).
```

---

## 3. Reference List Guidelines

### **Alphabetical Order**
References should be listed alphabetically by the first author's last name.

### **Digital Link Formatting**
All references should include clickable digital links when available:

```markdown
- Journal articles: Always include DOI link
- Conference papers: Include DOI when available
- Open access papers: Include direct PDF link
- Software repositories: Include GitHub link
- Technical documentation: Include official documentation URL
```

### **Hanging Indent Format**
```markdown
Craig, J. J. (2017). *Introduction to robotics: Mechanics and control* (4th ed.). Pearson. https://doi.org/10.1017/CBO9780511840232

Khalil, W., & Dombre, E. (2004). *Modeling, identification and control of robots*. Hermes Penton Science. https://doi.org/10.1201/9781420034516
```

---

## 4. Digital Object Identifiers (DOIs)

### **DOI Format Standards**
```markdown
Correct format: https://doi.org/10.1000/xyz123
Incorrect format: http://dx.doi.org/10.1000/xyz123
```

### **DOI Resolution**
All DOIs should resolve to the publisher's landing page, providing:
- Abstract of the work
- Full citation information
- Access options (open access or subscription)
- Related works and citations

### **DOI Lookup Resources**
- **CrossRef Metadata Search**: https://search.crossref.org/
- **DOI.org**: https://doi.org/
- **Google Scholar**: https://scholar.google.com/

---

## 5. Online Resource Citation

### **Web Pages and Online Articles**
```markdown
Author, A. A., & Author, B. B. (Year, Month Day). *Title of web page*. Site Name. URL Retrieved date
```

**Example:**
```markdown
Open Robotics. (2023, October 15). *ROS 2 documentation*. ROS.org. https://docs.ros.org/en/rolling/
```

### **Software Documentation**
```markdown
SoftwareName Team. (Year). *SoftwareName version X.X.X documentation* [Software manual]. URL
```

**Example:**
```markdown
OpenCV Team. (2023). *OpenCV 4.8.0 documentation* [Software manual]. https://docs.opencv.org/4.8.0/
```

### **Online Forums and Community Resources**
```markdown
Author, A. A. (Year, Month Day). *Title of post* [Forum post]. Forum Name. URL
```

---

## 6. Software and Code Citation

### **Software Packages**
```markdown
Author, A. A. (Year). *SoftwareName* (Version X.X.X) [Computer software]. Repository URL. https://doi.org/xxxxx
```

**Example:**
```markdown
Bradski, G. (2000). *OpenCV Library* (Version 4.8.0) [Computer software]. https://github.com/opencv/opencv
```

### **Code Examples and Tutorials**
```markdown
Author, A. A. (Year). *Title of code example* [Source code]. GitHub Repository. URL
```

**Example:**
```markdown
RoboticsLab. (2023). *PID controller implementation for mobile robots* [Source code]. https://github.com/roboticslab/pid-controller
```

### **Datasets**
```markdown
Creator, A. A. (Year). *Name of dataset* [Dataset]. Repository URL. https://doi.org/xxxxx
```

**Example:**
```markdown
Geiger, A., Lenz, P., & Urtasun, R. (2013). *KITTI Vision Benchmark Suite* [Dataset]. https://doi.org/10.5281/zenodo.3477991
```

---

## 7. Citation Management Tools

### **Recommended Tools**

#### **Zotero** (Free/Open Source)
- **Features**: Reference management, PDF annotation, collaboration
- **Integration**: Browser plugins, Word/LibreOffice plugins
- **Storage**: 300MB free, additional storage available
- **Website**: https://zotero.org/

#### **Mendeley** (Freemium)
- **Features**: Reference management, PDF reader, social networking
- **Integration**: Desktop application, web library
- **Storage**: 2GB free
- **Website**: https://www.mendeley.com/

#### **EndNote** (Commercial)
- **Features**: Professional reference management
- **Integration**: Microsoft Word, LaTeX
- **Storage**: Unlimited with subscription
- **Website**: https://endnote.com/

### **BibTeX Integration**

#### **BibTeX Entry Examples**
```bibtex
@article{lowe2004distinctive,
  title={Distinctive image features from scale-invariant keypoints},
  author={Lowe, David G},
  journal={International Journal of Computer Vision},
  volume={60},
  number={2},
  pages={91--110},
  year={2004},
  publisher={Springer},
  doi={10.1023/B:VISI.0000029664.99615.94}
}

@inproceedings{krizhevsky2012imagenet,
  title={Imagenet classification with deep convolutional neural networks},
  author={Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
  booktitle={Advances in Neural Information Processing Systems},
  pages={1097--1105},
  year={2012},
  doi={10.1145/3065386}
}
```

---

## 🔗 **Quick Reference Cheat Sheet**

### **Common Citation Templates**

| Source Type | In-Text | Reference List |
|-------------|---------|----------------|
| **Journal Article** | (Author, Year) | Author. (Year). Title. *Journal*, Volume(Issue), Pages. https://doi.org/xxx |
| **Conference Paper** | (Author, Year) | Author. (Year). *Title*. Conference. https://doi.org/xxx |
| **Book** | (Author, Year) | Author. (Year). *Title*. Publisher. https://doi.org/xxx |
| **Website** | (Author, Year) | Author. (Year). *Title*. Site. URL |
| **Software** | (Author, Year) | Author. (Year). *Software* [Computer software]. URL |

### **Citation Best Practices**

1. **Be Consistent**: Use the same format throughout your document
2. **Include DOIs**: Always include DOI links when available
3. **Verify Links**: Test all URLs to ensure they work
4. **Update Regularly**: Check for newer versions of software and papers
5. **Give Credit**: Always cite sources for ideas, code, and data

### **Common Citation Mistakes to Avoid**

1. ❌ Missing DOI links for journal articles
2. ❌ Using outdated URLs
3. ❌ Inconsistent author name formatting
4. ❌ Missing publication years
5. ❌ Not citing software and code libraries
6. ❌ Using short URLs that hide the destination
7. ❌ Forgetting to cite online resources and datasets

---

## 📚 **Citation Resources and Tools**

### **Online Citation Generators**
- **Cite This For Me**: https://www.citethisforme.com/
- **BibMe**: https://www.bibme.org/
- **Citation Machine**: https://www.citationmachine.net/

### **DOI Lookup Services**
- **CrossRef**: https://www.crossref.org/
- **DataCite**: https://datacite.org/
- **Zenodo**: https://zenodo.org/

### **Style Guides**
- **APA Style Official**: https://apastyle.apa.org/
- **Purdue OWL APA Guide**: https://owl.purdue.edu/owl/research_and_citation/apa_style/apa_formatting_and_style_guide/general_format.html
- **IEEE Editorial Style Manual**: https://ieeeauthorcenter.ieee.org/wp-content/uploads/IEEE-Reference-Guide.pdf

---

**🎯 Remember**: Proper citation is not just about avoiding plagiarism—it's about giving credit to original creators, enabling readers to find sources, and contributing to the scholarly conversation in robotics research.