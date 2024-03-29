//===-- tsd.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_TSD_H_
#define SCUDO_TSD_H_

#include "atomic_helpers.h"
#include "common.h"
#include "mutex.h"
#include "thread_annotations.h"

#include <limits.h> // for PTHREAD_DESTRUCTOR_ITERATIONS
#include <pthread.h>

// With some build setups, this might still not be defined.
#ifndef PTHREAD_DESTRUCTOR_ITERATIONS
#define PTHREAD_DESTRUCTOR_ITERATIONS 4
#endif

namespace scudo {

template <class Allocator> struct alignas(SCUDO_CACHE_LINE_SIZE) TSD {
  // TODO: Add thread-safety annotation on `Cache` and `QuarantineCache`.
  typename Allocator::CacheT Cache;
  typename Allocator::QuarantineCacheT QuarantineCache;
  using ThisT = TSD<Allocator>;
  u8 DestructorIterations = 0;

  void init(Allocator *Instance) NO_THREAD_SAFETY_ANALYSIS {
    DCHECK_EQ(DestructorIterations, 0U);
    DCHECK(isAligned(reinterpret_cast<uptr>(this), alignof(ThisT)));
    Instance->initCache(&Cache);
    DestructorIterations = PTHREAD_DESTRUCTOR_ITERATIONS;
  }

  inline bool tryLock() NO_THREAD_SAFETY_ANALYSIS {
    if (Mutex.tryLock()) {
      atomic_store_relaxed(&Precedence, 0);
      return true;
    }
    if (atomic_load_relaxed(&Precedence) == 0)
      atomic_store_relaxed(
          &Precedence,
          static_cast<uptr>(getMonotonicTime() >> FIRST_32_SECOND_64(16, 0)));
    return false;
  }
  inline void lock() NO_THREAD_SAFETY_ANALYSIS {
    atomic_store_relaxed(&Precedence, 0);
    Mutex.lock();
  }
  inline void unlock() NO_THREAD_SAFETY_ANALYSIS { Mutex.unlock(); }
  inline uptr getPrecedence() { return atomic_load_relaxed(&Precedence); }

  void commitBack(Allocator *Instance) ASSERT_CAPABILITY(Mutex) {
    Instance->commitBack(this);
  }

private:
  HybridMutex Mutex;
  atomic_uptr Precedence = {};
};

} // namespace scudo

#endif // SCUDO_TSD_H_
