while true; do
python3.6 manage_gpu.py --t 300 || break
sleep 1;
done