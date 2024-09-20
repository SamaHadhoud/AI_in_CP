#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

const long long MOD = 1000000007;

int N;
vector<vector<int>> adj;

void dfs(int u, vector<bool>& visit, int& farthest, int& maxd, int d) {
  visit[u] = true;
  if (d > maxd) {
    farthest = u;
    maxd = d;
  }
  for (int v : adj[u]) {
    if (!visit[v]) {
      dfs(v, visit, farthest, maxd, d + 1);
    }
  }
}

int solve() {
  cin >> N;
  adj.assign(N, {});
  for (int i = 1, p; i < N; i++) {
    cin >> p;
    p--;
    adj[i].push_back(p);
    adj[p].push_back(i);
  }
  vector<bool> visit(N, false);
  int farthest = 0, diameter = 0;
  dfs(0, visit, farthest, diameter, 0);
  visit.assign(N, false);
  dfs(farthest, visit, farthest, diameter, 0);
  int ans = 0;
  for (int i = 0; i <= diameter; i++) {
    ans = (ans + (N - i)) % MOD;
  }
  return ans;
}

int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  int T;
  cin >> T;
  for (int t = 1; t <= T; t++) {
    cout << "Case #" << t << ": " << solve() << endl;
  }
  return 0;
}
