#include <c10/core/SymFloat.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymIntNodeImpl.h>
#include <array>

namespace c10 {

static std::array<SymIntNode, 2> normalize_symints(SymInt a_, SymInt b_) {
  SymIntNode a, b;
  if (a_.is_symbolic())
    a = a_.toSymIntNodeImpl();
  if (b_.is_symbolic())
    b = b_.toSymIntNodeImpl();

  SymIntNodeImpl* common = a ? a.get() : b.get();
  // TODO: technically we need to check that the classes match
  if (!a) {
    a = common->wrap(a_.as_int_unchecked());
    a_.toSymInt(a); //
  }
  if (!b) {
    b = common->wrap(b_.as_int_unchecked());
    b_.toSymInt(b);
  }
  return {a, b};
}

SymIntNode SymInt::toSymIntNodeImpl() const {
  TORCH_CHECK(is_symbolic());
  return SymIntNode::reclaim_copy(toSymIntNodeImplUnowned());
}

c10::SymInt SymInt::toSymInt(SymIntNode sin_sp) {
  auto ptr = static_cast<uint64_t>(
      reinterpret_cast<uintptr_t>(static_cast<void*>(sin_sp.release())));
  auto rep = (ptr & ~MASK) | IS_SYM;
  return c10::SymInt(UNCHECKED, static_cast<int64_t>(rep));
}

int64_t SymInt::guard_int(const char* file, int64_t line) const {
  if (!is_symbolic()) {
    return data_;
  }
  SymIntNode a = toSymIntNodeImpl();
  return a->guard_int(file, line);
}

SymInt::operator SymFloat() const {
  if (!is_symbolic()) {
    return SymFloat(double(data_));
  }
  return SymFloat::toSymFloat(toSymIntNodeImpl()->sym_float());
}

SymInt SymInt::operator+(SymInt sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymInt(data_ + sci.data_);
  }
  auto res = normalize_symints(*this, sci);
  return SymInt::toSymInt(res[0]->add(res[1]));
}

SymInt SymInt::operator-(SymInt sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymInt(data_ - sci.data_);
  }
  auto res = normalize_symints(*this, sci);
  return SymInt::toSymInt(res[0]->sub(res[1]));
}

SymInt SymInt::operator*(SymInt sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymInt(data_ * sci.data_);
  }
  auto res = normalize_symints(*this, sci);
  return SymInt::toSymInt(res[0]->mul(res[1]));
}

SymInt SymInt::operator/(SymInt sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymInt(data_ / sci.data_);
  }
  auto res = normalize_symints(*this, sci);
  return SymInt::toSymInt(res[0]->floordiv(res[1]));
}

SymInt SymInt::operator%(SymInt sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymInt(data_ % sci.data_);
  }
  auto res = normalize_symints(*this, sci);
  return SymInt::toSymInt(res[0]->mod(res[1]));
}

bool SymInt::operator==(SymInt sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ == sci.data_;
  }
  auto res = normalize_symints(*this, sci);
  return res[0]->eq(res[1])->bool_();
}

bool SymInt::operator!=(SymInt sci) const {
  return !(*this == sci);
}

bool SymInt::operator<(SymInt sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ < sci.data_;
  }
  auto res = normalize_symints(*this, sci);
  return res[0]->lt(res[1])->bool_();
}

bool SymInt::operator<=(SymInt sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ <= sci.data_;
  }
  auto res = normalize_symints(*this, sci);
  return res[0]->le(res[1])->bool_();
}

bool SymInt::operator>(SymInt sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ > sci.data_;
  }
  auto res = normalize_symints(*this, sci);
  return res[0]->gt(res[1])->bool_();
}

bool SymInt::operator>=(SymInt sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ >= sci.data_;
  }
  auto res = normalize_symints(*this, sci);
  return res[0]->ge(res[1])->bool_();
}

SymInt SymInt::min(SymInt sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return std::min(data_, sci.data_);
  }
  auto res = normalize_symints(*this, sci);
  return SymInt::toSymInt(res[0]->min(res[1]));
}
SymInt SymInt::max(SymInt sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return std::max(data_, sci.data_);
  }
  auto res = normalize_symints(*this, sci);
  return SymInt::toSymInt(res[0]->max(res[1]));
}

void SymInt::operator*=(SymInt sci) {
  *this = *this * sci;
}

void SymInt::operator+=(SymInt sci) {
  *this = *this + sci;
}

bool SymInt::operator<(int64_t sci) const {
  return *this < c10::SymInt(sci);
}

bool SymInt::operator<=(int64_t sci) const {
  return *this <= c10::SymInt(sci);
}

bool SymInt::operator>(int64_t sci) const {
  return *this > c10::SymInt(sci);
}

bool SymInt::operator>=(int64_t sci) const {
  return *this >= c10::SymInt(sci);
}

bool SymInt::operator==(int64_t sci) const {
  return *this == c10::SymInt(sci);
}

bool SymInt::operator!=(int64_t sci) const {
  return *this != c10::SymInt(sci);
}

SymInt SymInt::operator*(int64_t sci) const {
  return *this * c10::SymInt(sci);
}

std::ostream& operator<<(std::ostream& os, SymInt s) {
  if (s.is_symbolic()) {
    os << s.toSymIntNodeImpl()->str();
  } else {
    os << s.as_int_unchecked();
  }
  return os;
}

SymInt operator-(SymInt s) {
  if (s.is_symbolic()) {
    return SymInt::toSymInt(s.toSymIntNodeImpl()->neg());
  } else {
    return SymInt(-s.as_int_unchecked());
  }
}

} // namespace c10
