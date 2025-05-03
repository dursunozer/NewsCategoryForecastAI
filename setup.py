from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="turkce_haber_siniflandirma",
    version="0.1.0",
    author="AI Proje Ekibi",
    author_email="ornek@email.com",
    description="Türkçe haber başlıklarını kategorilerine göre sınıflandıran yapay zeka modeli",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kullanici/turkce_haber_siniflandirma",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: Turkish",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "haber-siniflandirma=proje_baslat:main",
        ],
    },
) 