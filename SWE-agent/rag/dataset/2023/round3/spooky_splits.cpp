#include <iostream>
#include <set>
#include <vector>
using namespace std;

int N, M;
vector<vector<int>> adj;
vector<bool> visit;
vector<int> sizes;

void rec(int u) {
  sizes.back()++;
  visit[u] = true;
  for (int v : adj[u]) {
    if (!visit[v]) {
      rec(v);
    }
  }
}

int tgt;
vector<int> groups;
multiset<int> state;
set<multiset<int>> seen;

bool can_partition(int i) {
  if (i == (int)sizes.size()) {
    for (int sz : groups) {
      if (sz != tgt) {
        return false;
      }
    }
    return true;
  }
  if (seen.count(state)) {
    return false;
  }
  seen.insert(state);
  for (int &g : groups) {
    if (g + sizes[i] <= tgt) {
      state.erase(state.find(g));
      g += sizes[i];
      state.insert(g);
      if (can_partition(i + 1)) {
        return true;
      }
      state.erase(state.find(g));
      g -= sizes[i];
      state.insert(g);
    }
  }
  return false;
}

void solve() {
  cin >> N >> M;
  adj.assign(N, {});
  visit.assign(N, false);
  sizes.clear();
  for (int i = 0, a, b; i < M; i++) {
    cin >> a >> b;
    a--;
    b--;
    adj[a].push_back(b);
    adj[b].push_back(a);
  }
  for (int i = 0; i < N; i++) {
    if (!visit[i]) {
      sizes.push_back(0);
      rec(i);
    }
  }
  for (int K = 1; K <= N; K++) {
    if (N % K == 0) {
      tgt = N / K;
      groups.assign(K, 0);
      state.clear();
      for (int i = 0; i < K; i++) {
        state.insert(0);
      }
      seen.clear();
      if (can_partition(0)) {
        cout << " " << K;
      }
    }
  }
}

int main() {
  int T;
  cin >> T;
  for (int t = 1; t <= T; t++) {
    cout << "Case #" << t << ":";
    solve();
    cout << endl;
  }
  return 0;
}