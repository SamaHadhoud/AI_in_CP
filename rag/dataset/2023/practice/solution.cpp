#include <iostream>
#include <string>

std::string solve(int S, int D, int K) {
    int total_buns = 2 * S + 2 * D;
    int total_cheese = S + D;
    int total_patties = S + 2 * D;

    if (total_buns >= K && total_cheese >= K && total_patties >= K) {
        return "YES";
    } else {
        return "NO";
    }
}

int main() {
    int T;
    std::cin >> T;

    for (int i = 1; i <= T; ++i) {
        int S, D, K;
        std::cin >> S >> D >> K;

        std::string result = solve(S, D, K);
        std::cout << "Case #" << i << ": " << result << std::endl;
    }

    return 0;
}