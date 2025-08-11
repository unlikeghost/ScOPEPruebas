# ScOPE (SMILES Compression Property Estimator)

ScOPE is a novel parameter-free classification method that leverages compression algorithms to predict molecular properties directly from SMILES strings. Developed as an alternative to computationally expensive deep learning models, ScOPE provides interpretable results while requiring minimal resources - making it ideal for researchers and labs with limited computational access.

## 🧪 Scientific Innovation
- **First compression-based approach** for molecular property prediction
- **Eliminates traditional training phases** - works directly with known samples
- **Resource-efficient** - runs on standard CPUs with small datasets (<100 compounds)


## 🚀 Features
- Compression-based classification
- Specialized for chemical data (SMILES)
- Interactive notebook support
- High performance for large datasets

## 📦 Installation

### Recommended method (uv)
```bash
uv venv
uv pip install -e install -r pyproject.toml
uv pip install -e install -r pyproject.toml --extra notebooks
```

*Note: You need uv installed (pip install uv)*

### Alternative method (pip)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 🤝 Contributions
1. Report bugs or request features
2. Submit pull requests
3. Share use cases

## 📫 Contact
Jesús Alan Hernández Galván
✉️ alanhernandezgalvan@gmail.com

## 📄 License
MIT © 2025 Jesús Alan Hernández Galván