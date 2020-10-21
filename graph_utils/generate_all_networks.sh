echo "################"
echo "gaia"
python generate_networks.py gaia --experiment inaturalist --upload_capacity 1e10 --download_capacity 1e10
echo "################"
echo "amazon_us"
python generate_networks.py amazon_us --experiment inaturalist --upload_capacity 1e10 --download_capacity 1e10
echo "################"
echo "geantdistance"
python generate_networks.py geantdistance --experiment inaturalist --upload_capacity 1e10 --download_capacity 1e10
echo "################"
echo "ebone"
python generate_networks.py ebone --experiment inaturalist --upload_capacity 1e10 --download_capacity 1e10
echo "################"
echo "exodus"
python generate_networks.py exodus --experiment inaturalist --upload_capacity 1e10 --download_capacity 1e10