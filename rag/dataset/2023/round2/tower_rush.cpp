#include <cassert>
#include <iostream>
#include <vector>
using namespace std;

const int MAX = 1000010;
const int MOD = 1000000007;
using LL = long long;

class mint {
  LL val;

  mint& normalize(LL x) {
    val = x % MOD;
    if (val < 0) {
      val += MOD;
    }
    return *this;
  }

  static LL inverse(LL a) {
    LL u = 0, v = 1, m = MOD;
    while (a != 0) {
      LL t = m / a;
      m -= t * a;
      swap(a, m);
      u -= t * v;
      swap(u, v);
    }
    assert(m == 1);
    return u;
  }

 public:
  mint(LL x = 0) { normalize(x); }

  LL operator()() const { return val; }

  template <typename U>
  explicit operator U() const { return static_cast<U>(val); }

  mint& operator=(LL x) { return normalize(x); }
  mint& operator=(const mint& x) { val = x.val; return *this; }

  mint& operator+=(const mint& x) { return normalize(val + x.val); }
  mint& operator-=(const mint& x) { return normalize(val - x.val); }
  mint& operator*=(const mint& x) { return normalize(val * x.val); }
  mint& operator/=(const mint& x) { return *this *= mint(inverse(x.val)); }

  mint& operator+=(LL x) { return *this += mint(x); }
  mint& operator-=(LL x) { return *this -= mint(x); }
  mint& operator*=(LL x) { return *this *= mint(x); }
  mint& operator/=(LL x) { return *this /= mint(x); }

  mint& operator++() { return *this += 1; }
  mint& operator--() { return *this -= 1; }
  mint operator++(int) { mint z(*this); ++*this; return z; }
  mint operator--(int) { mint z(*this); --*this; return z; }

  friend mint operator+(mint x, const mint& y) { return x += y; }
  friend mint operator*(mint x, const mint& y) { return x *= y; }
  friend mint operator-(mint x, const mint& y) { return x -= y; }
  friend mint operator/(mint x, const mint& y) { return x /= y; }

  friend mint operator+(mint x, LL y) { return x += y; }
  friend mint operator*(mint x, LL y) { return x *= y; }
  friend mint operator-(mint x, LL y) { return x -= y; }
  friend mint operator/(mint x, LL y) { return x /= y; }

  friend mint operator+(LL x, mint y) { return y += x; }
  friend mint operator*(LL x, mint y) { return y *= x; }
  friend mint operator-(LL x, const mint& y) { mint z(x); return z -= y; }
  friend mint operator/(LL x, const mint& y) { mint z(x); return z /= y; }

  bool operator <(const mint& x) const { return val < x.val; }
  bool operator==(const mint& x) const { return val == x.val; }
  bool operator >(const mint& x) const { return val > x.val; }
  bool operator!=(const mint& x) const { return val != x.val; }
  bool operator<=(const mint& x) const { return val <= x.val; }
  bool operator>=(const mint& x) const { return val >= x.val; }

  bool operator <(LL x) const { return val < x; }
  bool operator==(LL x) const { return val == x; }
  bool operator >(LL x) const { return val > x; }
  bool operator!=(LL x) const { return val != x; }
  bool operator<=(LL x) const { return val <= x; }
  bool operator>=(LL x) const { return val >= x; }
};

class factorial : vector<mint> {
  void lazy_eval(int n) {
    for (LL p = size(); n >= p; ++p) {
      push_back(back() * p);
    }
  }

 public:
  factorial() { push_back(1); }

  mint choose(LL n, LL k) {
    if (n < 0 || k < 0 || n < k) {
      return 0;
    }
    if (k == 0 || k == n) {
      return 1;
    }
    if (n >= MOD && n <= k + MOD - 1) {
      return 0;
    }
    lazy_eval(n);
    return at(n) / (at(k) * at(n - k));
  }

  mint operator[](LL n) {
    lazy_eval(n);
    return at(n);
  }
};

factorial F;
vector<vector<LL>> divisors(MAX);
vector<LL> mu(MAX);

void build_mu() {
  vector<bool> B(MAX);
  vector<LL> primes;
  mu[1] = 1LL;
  for (LL i = 2; i < MAX; i++) {
    if (!B[i]) {
      primes.push_back(i);
      mu[i] = -1;
    }
    for (auto& p : primes) {
      LL k = (LL)i * p;
      if (k > MAX) {
        break;
      }
      B[k] = 1;
      if (i % p != 0) {
        mu[k] = -mu[i];
      } else {
        mu[k] = 0;
        break;
      }
    }
  }
}

LL solve() {
  int N, K, D;
  cin >> N >> K >> D;
  vector<int> H(N), freq(MAX);
  for (int i = 0; i < N; i++) {
    cin >> H[i];
    for (LL d : divisors[H[i]]) {
      freq[d]++;
    }
  }
  mint ans = 0;
  for (LL d : divisors[D]) {
    for (LL j = 0; j < MAX; j++) {
      LL i = (LL)d * j;
      if (i > MAX) {
        break;
      }
      ans += F.choose(freq[i], K) * mu[j];
    }
  }
  return (LL)(ans * F[K]);
}

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(nullptr);

  for (int i = 1; i < MAX; i++) {
    for (int j = i; j < MAX; j += i) {
      divisors[j].push_back(i);
    }
  }
  build_mu();

  int T;
  cin >> T;
  for (int t = 1; t <= T; t++) {
    cout << "Case #" << t << ": " << solve() << endl;
  }
  return 0;
}