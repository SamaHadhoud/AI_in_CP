#include <algorithm>
#include <iostream>
#include <tuple>
#include <vector>
using namespace std;

using int64 = long long;

int N, M;
vector<int64> P;
vector<pair<int, int>> dsu_lists;
vector<int> dsu_par;
vector<int64> dsu_w, ans;

inline int get_parent(int x) {
  return x == dsu_par[x] ? x : (dsu_par[x] = get_parent(dsu_par[x]));
}

void unite(int x, int y) {
  if (P[x] < P[y]) {
    swap(x, y);
  }
  if (P[x] > dsu_w[y]) {
    for (int ind = y;;) {
      ans[ind] += max(0LL, P[x] - dsu_w[y] - ans[ind]);
      if (ind == dsu_lists[ind].first) {
        break;
      }
      ind = dsu_lists[ind].first;
    }
  }
  dsu_lists[dsu_lists[x].second].first = y;
  dsu_lists[x].second = dsu_lists[y].second;
  dsu_w[x] += dsu_w[y];
  dsu_par[y] = dsu_par[x];
}

int64 solve() {
  cin >> N >> M;
  P.resize(N);
  for (int i = 0; i < N; i++) {
    cin >> P[i];
  }
  vector<pair<int, int>> E(M);
  vector<tuple<int64, int64, int>> order;
  for (int i = 0; i < M; i++) {
    cin >> E[i].first >> E[i].second;
    if (P[--E[i].first] < P[--E[i].second]) {
      swap(E[i].first, E[i].second);
    }
    order.emplace_back(P[E[i].first], P[E[i].second], i);
  }
  sort(order.begin(), order.end());
  vector<pair<int, int>> E_buf(M);
  for (int i = 0; i < M; i++) {
    E_buf[i] = E[get<2>(order[i])];
  }
  E = E_buf;
  ans.assign(N, 0LL);
  dsu_lists.resize(N);
  dsu_w.assign(N, 0LL);
  dsu_par.resize(N);
  for (int i = 0; i < N; i++) {
    dsu_par[i] = i;
    dsu_w[i] = P[i];
    dsu_lists[i] = make_pair(i, i);
  }
  for (int i = 0; i < M; i++) {
    int x = get_parent(E[i].first);
    int y = get_parent(E[i].second);
    if (x == y) {
      continue;
    }
    unite(x, y);
  }
  int64 res = 0LL;
  for (int i = 0; i < N; i++) {
    res += ans[i];
  }
  return res;
}

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  int T;
  cin >> T;
  for (int t = 1; t <= T; t++) {
    cout << "Case #" << t << ": " << solve() << endl;
  }
  return 0;
}
