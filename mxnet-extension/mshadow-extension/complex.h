/*!
 *  Copyright (c) 2016 by Contributors
 * \file complex.h
 * \brief support for complex operations
 * \author Xingjian Shi
 */
#ifndef MSHADOW_EXTENSION_COMPLEX_H_
#define MSHADOW_EXTENSION_COMPLEX_H_
#include <algorithm>
#include "../extension.h"

namespace mshadow {
namespace op {
  struct complex_mul{
    /*! \brief map a_real, a_imag, b_real, b_imag to result using defined operation */
    template<typename DType>
    MSHADOW_XINLINE static DType RealMap(DType a_real, DType a_imag, DType b_real, DType b_imag) {
      return a_real * b_real - a_imag * b_imag;
    }
    template<typename DType>
    MSHADOW_XINLINE static DType ImagMap(DType a_real, DType a_imag, DType b_real, DType b_imag) {
      return a_real * b_imag + b_real * a_imag;
    }
  };
  struct complex_div{
    /*! \brief map a_real, a_imag, b_real, b_imag to result using defined operation */
    template<typename DType>
    MSHADOW_XINLINE static DType RealMap(DType a_real, DType a_imag, DType b_real, DType b_imag) {
      return (a_real * b_real + a_imag * b_imag) / (b_real * b_real + b_imag * b_imag);
    }
    template<typename DType>
    MSHADOW_XINLINE static DType ImagMap(DType a_real, DType a_imag, DType b_real, DType b_imag) {
      return (a_real * b_imag - b_real * a_imag) / (b_real * b_real + b_imag * b_imag);
    }
  };
}

namespace expr {
//--------------------
// ComplexBinaryMapExp
//--------------------
  /*!
* \brief binary map expression lhs [op] rhs where lhs and rhs are complex tensors
* \tparam OP operator
* \tparam TA type of lhs
* \tparam TB type of rhs
* \tparam etype expression type, sa namespace::type
*/
template<typename OP, typename TA, typename TB, typename DType, int etype>
struct ComplexBinaryMapExp : public Exp<ComplexBinaryMapExp<OP, TA, TB, DType, etype>,
  DType, etype> {
  /*! \brief left operand */
  const TA &lhs_;
  /*! \brief right operand */
  const TB &rhs_;
  /*! \brief constructor */
  explicit ComplexBinaryMapExp(const TA &lhs, const TB &rhs)
    :lhs_(lhs), rhs_(rhs) {}
};

//-------------------
// ComplexConjExp
//-------------------
/*!
* \brief compute conj(src) where src is a complex tensor
* \tparam TA type of src
* \tparam etype expression type, sa namespace::type
*/
template<typename TA, typename DType, int etype>
struct ComplexConjugateExp : public Exp<ComplexConjugateExp<TA, DType, etype>,
  DType, etype> {
  /*! \brief source expression */
  const TA &src_;
  /*! \brief constructor */
  explicit ComplexConjugateExp(const TA &src) : src_(src) {}
};

//-------------------
// ComplexExchangeExp
//-------------------
/*!
* \brief compute the complex conjugate of src
* \tparam TA type of src
* \tparam etype expression type, sa namespace::type
*/
template<typename TA, typename DType, int etype>
struct ComplexExchangeExp : public Exp<ComplexExchangeExp<TA, DType, etype>,
  DType, etype> {
  /*! \brief source expression */
  const TA &src_;
  /*! \brief constructor */
  explicit ComplexExchangeExp(const TA &src) : src_(src) {}
};



template<typename OP, typename TA, typename TB, typename DType, int ta, int tb>
inline ComplexBinaryMapExp<OP, TA, TB, DType, (ta | tb | type::kMapper)>
ComplexF(const Exp<TA, DType, ta> &lhs, const Exp<TB, DType, tb> &rhs) {
  return ComplexBinaryMapExp<OP, TA, TB, DType,
    (ta | tb | type::kMapper)>(lhs.self(), rhs.self());
}

/*!
* \brief conj Negation the imaginary part of A where A is a complex tensor
* \param src source tensor
* \tparam e1 type of source expression
*/
template<typename SrcExp, typename DType, int e1>
inline ComplexConjugateExp<SrcExp, DType, (e1|type::kMapper)>
conj(const Exp<SrcExp, DType, e1> &src) {
  return ComplexConjugateExp<SrcExp, DType, (e1|type::kMapper)>(src.self());
}

/*!
* \brief complex_exchange Exchange the real and imaginary part of A where A is a complex 4D tensors
* \param src source tensor
* \tparam e1 type of source expression
*/
template<typename SrcExp, typename DType, int e1>
inline ComplexExchangeExp<SrcExp, DType, (e1|type::kMapper)>
complex_exchange(const Exp<SrcExp, DType, e1> &src) {
  return ComplexExchangeExp<SrcExp, DType, (e1|type::kMapper)>(src.self());
}



template<int dim, typename OP, typename TA, typename TB,
  typename DType, int etype>
struct ShapeCheck<dim, ComplexBinaryMapExp<OP, TA, TB, DType, etype> > {
  inline static Shape<dim>
    Check(const ComplexBinaryMapExp<OP, TA, TB, DType, etype> &t) {
    Shape<dim> shape1 = ShapeCheck<dim, TA>::Check(t.lhs_);
    Shape<dim> shape2 = ShapeCheck<dim, TB>::Check(t.rhs_);
    if (shape1[0] == 0) return shape2;
    if (shape2[0] == 0) return shape1;
    CHECK_EQ(shape1, shape2) << "ComplexBinaryMapExp: Shapes of operands are not the same";
    CHECK_EQ(shape1[dim - 1] % 2, 0) << "ComplexBinaryMapExp: Shape of the last dimension is not even. "
      "We must have real + imaginary.";
    return shape1;
  }
};

template<int dim, typename TA, typename DType, int etype>
struct ShapeCheck<dim, ComplexConjugateExp<TA, DType, etype> > {
  inline static Shape<dim> Check(const ComplexConjugateExp<TA, DType, etype> &t) {
    Shape<dim> s = ShapeCheck<dim, TA>::Check(t.src_);
    CHECK_EQ(s[dim - 1] % 2, 0) << "ComplexConjExp: Shape of the last dimension is not even. "
      "We must have real + imaginary.";
    return s;
  }
};

template<int dim, typename TA, typename DType, int etype>
struct ShapeCheck<dim, ComplexExchangeExp<TA, DType, etype> > {
  inline static Shape<dim> Check(const ComplexExchangeExp<TA, DType, etype> &t) {
    Shape<dim> s = ShapeCheck<dim, TA>::Check(t.src_);
    CHECK_EQ(s[dim - 1] % 2, 0) << "ComplexExchangeExp: Shape of the last dimension is not even. "
      "We must have real + imaginary.";
      return s;
  }
};



// complex binary expression
template<typename OP, typename TA, typename TB, int etype, typename DType>
class Plan<ComplexBinaryMapExp<OP, TA, TB, DType, etype>, DType> {
public:
  explicit Plan(const Plan<TA, DType> &lhs, const Plan<TB, DType> &rhs)
    : lhs_(lhs), rhs_(rhs) {}
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    const index_t base_x = static_cast<index_t>(x / 2) * 2;
    if (x % 2 == 0) {
      return OP::RealMap(lhs_.Eval(y, base_x), lhs_.Eval(y, base_x + 1), rhs_.Eval(y, base_x), rhs_.Eval(y, base_x + 1));
    } else{
      return OP::ImagMap(lhs_.Eval(y, base_x), lhs_.Eval(y, base_x + 1), rhs_.Eval(y, base_x), rhs_.Eval(y, base_x + 1));
    }
  }

private:
  Plan<TA, DType> lhs_;
  Plan<TB, DType> rhs_;
};

// complex conjugate expression
template<typename TA, int etype, typename DType>
class Plan<ComplexConjugateExp<TA, DType, etype>, DType> {
public:
  explicit Plan(const Plan<TA, DType> &src) : src_(src) {}
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    const index_t base_x = static_cast<index_t>(x / 2) * 2;
    if (0 == x % 2) {
      return src_.Eval(y, base_x);
    }
    else {
      return - src_.Eval(y, base_x + 1);
    }
  }

private:
  Plan<TA, DType> src_;
};

// complex exchange expression
template<typename TA, int etype, typename DType>
class Plan<ComplexExchangeExp<TA, DType, etype>, DType> {
public:
  explicit Plan(const Plan<TA, DType> &src) : src_(src) {}
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    const index_t base_x = static_cast<index_t>(x / 2) * 2;
    if (0 == x % 2) {
      return src_.Eval(y, base_x + 1);
    }
    else {
      return src_.Eval(y, base_x);
    }
  }

private:
  Plan<TA, DType> src_;
};



template<typename OP, typename TA, typename TB, typename DType, int etype>
inline Plan<ComplexBinaryMapExp<OP, TA, TB, DType, etype>, DType>
MakePlan(const ComplexBinaryMapExp<OP, TA, TB, DType, etype> &e) {
  return Plan<ComplexBinaryMapExp<OP, TA, TB, DType, etype>,
    DType>(MakePlan(e.lhs_), MakePlan(e.rhs_));
}

template<typename TA, typename DType, int etype>
inline Plan<ComplexConjugateExp<TA, DType, etype>, DType>
MakePlan(const ComplexConjugateExp<TA, DType, etype> &e) {
  return Plan<ComplexConjugateExp<TA, DType, etype>, DType>(MakePlan(e.src_));
}

template<typename TA, typename DType, int etype>
inline Plan<ComplexExchangeExp<TA, DType, etype>, DType>
MakePlan(const ComplexExchangeExp<TA, DType, etype> &e) {
  return Plan<ComplexExchangeExp<TA, DType, etype>, DType>(MakePlan(e.src_));
}



template<typename OP, typename TA, typename TB, typename DType, int etype>
struct ExpInfo<ComplexBinaryMapExp<OP, TA, TB, DType, etype> > {
  static const int kDimLhs = ExpInfo<TA>::kDim;
  static const int kDimRhs = ExpInfo<TB>::kDim;
  static const int kDim = (kDimLhs >= 0 && kDimRhs >= 0) ? \
    (kDimLhs == 0 ? \
  kDimRhs : \
            ((kDimRhs == 0 || kDimLhs == kDimRhs) ? kDimLhs : -1)) : -1;
  static const int kDevMask = ExpInfo<TA>::kDevMask & ExpInfo<TB>::kDevMask;
};

template<typename TA, typename DType, int etype>
struct ExpInfo<ComplexConjugateExp<TA, DType, etype> > {
  static const int kDim = ExpInfo<TA>::kDim;
  static const int kDevMask = ExpInfo<TA>::kDevMask;
};

template<typename TA, typename DType, int etype>
struct ExpInfo<ComplexExchangeExp<TA, DType, etype> > {
  static const int kDim = ExpInfo<TA>::kDim;
  static const int kDevMask = ExpInfo<TA>::kDevMask;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_COMPLEX_H_
