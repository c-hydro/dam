# Installation

### Prerequisites

- Python 3.12 or higher
- System dependencies: GDAL, PROJ, GEOS (for geospatial packages)

On Ubuntu/Debian:
```bash
sudo apt-get install gdal-bin libgdal-dev libproj-dev libgeos-dev
```

### Install from GitHub

dam depends on d3tools, which must be installed first:

```bash
pip install git+https://github.com/c-hydro/d3tools@main
pip install git+https://github.com/c-hydro/dam@main
```

dam is often used in conjuction with the packages door and dryes, also developed by CIMA foundation, you can install them as well from github:

```bash
pip install git+https://github.com/c-hydro/door@main
pip install git+https://github.com/c-hydro/dryes@main
```

### Install from GitHub using uv (recommended)

```bash
uv pip install git+https://github.com/c-hydro/d3tools@main
uv pip install git+https://github.com/c-hydro/dam@main

# optionally
uv pip install git+https://github.com/c-hydro/door@main
uv pip install git+https://github.com/c-hydro/dryes@main
```