import re
import setuptools
from pathlib import Path

def get_sf_version():
    init_path = Path("sf/__init__.py")
    text = init_path.read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*[\'"]([^\'"]+)[\'"]', text)
    if not m:
        raise RuntimeError(
            f"Cannot find __version__ in {init_path}. "
            "Please add a line like: __version__ = '0.1.0'"
        )
    return m.group(1)

def get_long_description():
    return Path("README.md").read_text(encoding="utf-8")

setuptools.setup(
    name="StrAP",
    version=get_sf_version(),
    url="",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    setup_requires=["wheel"],
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
