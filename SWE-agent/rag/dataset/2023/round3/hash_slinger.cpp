#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

const int MAX = 1 << 13;

int solve() {
  int N, K;
  cin >> N >> K;
  vector<int> A(N);
  for (int i = 0; i < N; i++) {
    cin >> A[i];
    A[i] %= K;
  }
  vector<int> first_sofar(K, N + 1);
  vector<vector<int>> first(N, vector<int>(K));
  for (int i = N - 1; i >= 0; --i) {
    int d = 0;
    for (int j = i; j < N; ++j) {
      d = (d + A[j]) % K;
      first_sofar[d] = min(first_sofar[d], j + 1);
    }
    first[i] = first_sofar;
  }
  vector<int> dist(MAX, N + 1);
  vector<bool> visit(MAX);
  dist[0] = 0;
  for (;;) {
    int x = -1;
    for (int j = 0; j < MAX; ++j) {
      if (!visit[j] && dist[j] < N && (x == -1 || dist[x] > dist[j])) {
        x = j;
      }
    }
    if (x == -1) {
      break;
    }
    visit[x] = 1;
    for (int d = 0; d < K; d++) {
      dist[x ^ d] = min(dist[x ^ d], first[dist[x]][d]);
    }
  }
  int res = 0;
  for (int i = 0; i < MAX; ++i) {
    res += (dist[i] != N + 1);
  }
  return res;

}

int main() {
  int T;
  cin >> T;
  for (int t = 1; t <= T; t++) {
    cout << "Case #" << t << ": " << solve() << endl;
  }
  return 0;
}
