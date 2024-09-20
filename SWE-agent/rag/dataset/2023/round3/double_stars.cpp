#include <iostream>
#include <map>
#include <vector>
using namespace std;

// Solution:
// - For each edge u-v, find d(u, v) = longest arm length starting with u-v-..
// - For each edge u-v, consider all double stars (x, y) with u-v as the center.
// - Iterate through f[u] and f[v] with 2-pointer, with d' initially 1:
//   - For the first (d, x), sum min(degree(u) - 1, degree(v) - 1) for every
//     [d', d] (those d's, we can make 2-star with that number of arms).
//   - Remove the first and decrease x from degree(u) or degree(v), depending
//     on what the pair belongs to.
//   - Set d' = d + 1

vector<vector<int>> adj;
vector<vector<int>> d;  // d[u][i] = greatest distance reachable from adj[u][i]
vector<map<int, int>> f;  // f[u] = map of (d, x), representing d is the
                          // greatest distance of an arm (u, v) for x edges v

int pre(int u, int par = -1, int l = 0) {
  int mai = 0;
  for (int i = 0; i < (int)adj[u].size(); ++i) {
    int v = adj[u][i];
    d[u].push_back(-1);
    if (v == par) {
      continue;
    }
    int ret = pre(v, u);
    mai = max(mai, ret);
    d[u][i] = ret;
  }
  return 1 + mai;
}

void pre2(int u, int par = -1, int farthest_up = 0) {
  vector<int> pref(adj[u].size()), suf(adj[u].size());
  for (int i = 0; i < (int)adj[u].size(); ++i) {
    pref[i] = max(i ? pref[i - 1] : 0, d[u][i]);
  }
  for (int i = (int)adj[u].size() - 1; i >= 0; --i) {
    suf[i] = max(i + 1 < (int)adj[u].size() ? suf[i + 1] : 0, d[u][i]);
  }
  for (int i = 0; i < (int)adj[u].size(); ++i) {
    int v = adj[u][i];
    if (v == par) {
      d[u][i] = farthest_up;
      continue;
    }
    // Farthest going down here.
    int tmp = 1 + max(i ? pref[i - 1] : 0,
                      i + 1 < (int)adj[u].size() ? suf[i + 1] : 0);
    pre2(v, u, max(1 + farthest_up, tmp));
  }
}

long long rec(int u, int par = -1) {
  long long ans = 0;
  for (int i = 0; i < (int)adj[u].size(); ++i) {
    int v = adj[u][i];
    if (v == par) {
      // Calculate for DST centered at (u, par).
      f[u][d[u][i]]--;

      auto it1 = f[v].begin();
      auto it2 = f[u].begin();
      int tot1 = (int)adj[v].size() - 1;
      int tot2 = (int)adj[u].size() - 1;

      int l = 1;
      while (it1 != f[v].end() || it2 != f[u].end()) {
        if (it1 != f[v].end() &&
            (it2 == f[u].end() || it1->first < it2->first)) {
          ans += (long long)(it1->first - l + 1) * min(tot1, tot2);
          tot1 -= it1->second;
          l = it1->first + 1;
          it1++;
        } else {
          ans += (long long)(it2->first - l + 1) * min(tot1, tot2);
          tot2 -= it2->second;
          l = it2->first + 1;
          it2++;
        }
      }
      // assert(tot1 == 0 && tot2 == 0);
      f[u][d[u][i]]++;
      continue;
    }
    f[u][d[u][i]]--;
    ans += rec(v, u);
    f[u][d[u][i]]++;
  }
  return ans;
}

long long solve() {
  int N;
  cin >> N;
  adj.assign(N + 1, {});
  d.assign(N + 1, {});
  f.assign(N + 1, {});
  for (int i = 0, p; i < N - 1; i++) {
    cin >> p;
    adj[i + 2].push_back(p);
    adj[p].push_back(i + 2);
  }
  pre(1);
  pre2(1);
  for (int u = 1; u <= N; ++u) {
    for (auto dd : d[u]) {
      f[u][dd]++;
    }
  }
  return rec(1);
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