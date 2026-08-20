# Input parameters for optimization

# RGB color space selection
m = ["sRGB"]  # Options from rgb working space.xlsx

# Species selection
species = "Human"
# species = ["Homo sapiens melanopic"]  # Options from spectral curves travis.xlsx
# species = ["Felis catus"]  # Options from spectral curves travis.xlsx
# species = ["Chelonia mydas"]  # Options from spectral curves travis.xlsx

# Lighting setup
daylight = ["CIE D Series 5000 K"]  # Options from Daylight Sources.xlsx
# daylight = ["CIE D Series 7500 K"]  # Options from Daylight Sources.xlsx
sources = ["telelumen"] #"Lighting Lab"] #"Rubik Condensed"]   # Options from Light Sources.xlsx
# ref_source = ["Sample LED 4000 K"]
iesfile = ["telelumen"] # or rubik or white
ref_source = ["reference"]