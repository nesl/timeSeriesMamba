h_values=("AZPS" "CISO" "NYIS" "FR" "PJM" "SWPP" "TVA")

for h in "${h_values[@]}"; do
    ./scripts/testCarbonCastTransfer.sh -l 0 -d 32 -e 10 -c 100 -f 1 -t 100 -r 0 -m GPT2 -n 130m -g 3 -p 13321 -h "$h"
done
