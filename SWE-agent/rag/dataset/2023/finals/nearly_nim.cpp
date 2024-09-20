#include <iostream>
#include <vector>
using namespace std;

int solve() {
  int N;
  cin >> N;
  vector<int> A(N);
  for (int i = 0; i < N; i++) {
    cin >> A[i];
  }
  int ans = 0;
  vector<int> left(N + 1), right(N + 1);
  for (int i = 0; i < N; i++) {
    left[i + 1] = max(0, A[i] - left[i]);
  }
  for (int i = N - 1; i >= 0; i--) {
    right[i - 1] = max(0, A[i] - right[i]);
  }
  for (int i = 0; i < N; i++) {
    ans += max(0, A[i] - left[i] - right[i]);
  }
  return ans;
}

int main() {
  int T;
  cin >> T;
  for (int t = 1; t <= T; t++) {
    cout << "Case #" << t << ": " << solve() << endl;
  }
  return 0;
}
