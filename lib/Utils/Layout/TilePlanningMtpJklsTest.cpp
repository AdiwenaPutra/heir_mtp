#include "lib/Utils/Layout/TilePlanningMtpJkls.h"

#include <cstdint>
#include <limits>
#include <set>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Layout/Utils.h"
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace {

// A handwritten oracle for the valid (unpadded) extent of tile `tile` along
// a dimension of size `dimSize`, tiled by `tileSize` -- independent of
// TilePlanningMtpJkls.cpp's own validExtent so a bug shared between the
// implementation and this helper would not go unnoticed.
int64_t oracleValidExtent(int64_t dimSize, int64_t tile, int64_t tileSize) {
  int64_t remaining = dimSize - tile * tileSize;
  return remaining < tileSize ? remaining : tileSize;
}

// ---------------------------------------------------------------------
// Basic construction and full-plan structural checks
// ---------------------------------------------------------------------

TEST(TilePlanningMtpJklsTest, SingleTileNoTiling) {
  // M=K=N=mu: I=Q=J=1, exactly one task, no boundary.
  auto plan = computeGemmTilePlan(/*M=*/2, /*K=*/2, /*N=*/2, /*mu=*/2,
                                  /*minSlotCount=*/8);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(plan->I, 1);
  EXPECT_EQ(plan->Q, 1);
  EXPECT_EQ(plan->J, 1);
  EXPECT_EQ(plan->taskCount, 1);
  ASSERT_EQ(plan->tasks.size(), 1u);
  const GemmTask& task = plan->tasks[0];
  EXPECT_EQ(task.i, 0);
  EXPECT_EQ(task.j, 0);
  EXPECT_EQ(task.q, 0);
  EXPECT_EQ(task.validRows, 2);
  EXPECT_EQ(task.validContraction, 2);
  EXPECT_EQ(task.validColumns, 2);
  EXPECT_TRUE(succeeded(verifyGemmTilePlan(*plan)));
}

TEST(TilePlanningMtpJklsTest, MultipleOutputTilesB2Shape) {
  // A:2x2, B:2x4, mu=2 -- Phase 8's B2 fixture shape: I=1, Q=1, J=2.
  auto plan = computeGemmTilePlan(/*M=*/2, /*K=*/2, /*N=*/4, /*mu=*/2,
                                  /*minSlotCount=*/8);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(plan->I, 1);
  EXPECT_EQ(plan->Q, 1);
  EXPECT_EQ(plan->J, 2);
  EXPECT_EQ(plan->taskCount, 2);
  ASSERT_EQ(plan->tasks.size(), 2u);
  EXPECT_TRUE(succeeded(verifyGemmTilePlan(*plan)));

  // Both tasks share q=0 and i=0 (lhs A[0,0] is reused), but have distinct
  // j and distinct destTileId/rhsTileId.
  EXPECT_EQ(plan->tasks[0].i, 0);
  EXPECT_EQ(plan->tasks[0].j, 0);
  EXPECT_EQ(plan->tasks[1].i, 0);
  EXPECT_EQ(plan->tasks[1].j, 1);
  EXPECT_EQ(plan->tasks[0].lhsTileId, plan->tasks[1].lhsTileId);
  EXPECT_NE(plan->tasks[0].rhsTileId, plan->tasks[1].rhsTileId);
  EXPECT_NE(plan->tasks[0].destTileId, plan->tasks[1].destTileId);
}

TEST(TilePlanningMtpJklsTest, MultipleContractionTilesB3Shape) {
  // A:2x4, B:4x2, mu=2 -- Phase 8's B3 fixture shape: I=1, Q=2, J=1.
  auto plan = computeGemmTilePlan(/*M=*/2, /*K=*/4, /*N=*/2, /*mu=*/2,
                                  /*minSlotCount=*/8);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(plan->I, 1);
  EXPECT_EQ(plan->Q, 2);
  EXPECT_EQ(plan->J, 1);
  EXPECT_EQ(plan->taskCount, 1);
  ASSERT_EQ(plan->tasks.size(), 2u);
  EXPECT_TRUE(succeeded(verifyGemmTilePlan(*plan)));

  // Exactly one destination tile; both tasks accumulate into it, in q order.
  std::vector<GemmTask> chain = plan->reductionChainForDest(0, 0);
  ASSERT_EQ(chain.size(), 2u);
  EXPECT_EQ(chain[0].q, 0);
  EXPECT_EQ(chain[1].q, 1);
  EXPECT_NE(chain[0].lhsTileId, chain[1].lhsTileId);
  EXPECT_NE(chain[0].rhsTileId, chain[1].rhsTileId);
}

TEST(TilePlanningMtpJklsTest, RectangularBoundaryB6Shape) {
  // A:3x5, B:5x4, mu=2 -- Phase 8's mandatory B6 fixture shape:
  // I=2, Q=3, J=2, taskCount=4, tasks=12. Row tile i=1 and contraction tile
  // q=2 are boundary (valid extent 1 of mu=2); no column boundary (N=4).
  auto plan = computeGemmTilePlan(/*M=*/3, /*K=*/5, /*N=*/4, /*mu=*/2,
                                  /*minSlotCount=*/32);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(plan->I, 2);
  EXPECT_EQ(plan->Q, 3);
  EXPECT_EQ(plan->J, 2);
  EXPECT_EQ(plan->taskCount, 4);
  EXPECT_EQ(plan->tasks.size(), 12u);
  EXPECT_TRUE(succeeded(verifyGemmTilePlan(*plan)));

  for (const GemmTask& task : plan->tasks) {
    int64_t expectedRows = task.i == 1 ? 1 : 2;
    int64_t expectedK = task.q == 2 ? 1 : 2;
    EXPECT_EQ(task.validRows, expectedRows);
    EXPECT_EQ(task.validContraction, expectedK);
    EXPECT_EQ(task.validColumns, 2);  // N=4 divides mu=2 evenly
  }
}

TEST(TilePlanningMtpJklsTest, ColumnBoundaryShape) {
  // A:4x4, B:4x3, mu=2: I=2, Q=2, J=2, column tile j=1 is boundary (valid
  // extent 1 of mu=2); no row or contraction boundary.
  auto plan = computeGemmTilePlan(/*M=*/4, /*K=*/4, /*N=*/3, /*mu=*/2,
                                  /*minSlotCount=*/16);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(plan->I, 2);
  EXPECT_EQ(plan->Q, 2);
  EXPECT_EQ(plan->J, 2);
  EXPECT_TRUE(succeeded(verifyGemmTilePlan(*plan)));

  for (const GemmTask& task : plan->tasks) {
    EXPECT_EQ(task.validRows, 2);
    EXPECT_EQ(task.validContraction, 2);
    EXPECT_EQ(task.validColumns, task.j == 1 ? 1 : 2);
  }
}

TEST(TilePlanningMtpJklsTest, TileLargerThanDimensionIsNotRejected) {
  // mu larger than a matrix dimension is legal, not an error: mu=4, M=3
  // gives I=1 with a single boundary tile of valid extent 3.
  auto plan = computeGemmTilePlan(/*M=*/3, /*K=*/3, /*N=*/3, /*mu=*/4,
                                  /*minSlotCount=*/32);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(plan->I, 1);
  EXPECT_EQ(plan->Q, 1);
  EXPECT_EQ(plan->J, 1);
  ASSERT_EQ(plan->tasks.size(), 1u);
  EXPECT_EQ(plan->tasks[0].validRows, 3);
  EXPECT_EQ(plan->tasks[0].validContraction, 3);
  EXPECT_EQ(plan->tasks[0].validColumns, 3);
  EXPECT_TRUE(succeeded(verifyGemmTilePlan(*plan)));
}

// ---------------------------------------------------------------------
// Coverage, uniqueness, bounds, ordering (property-style, over one plan)
// ---------------------------------------------------------------------

TEST(TilePlanningMtpJklsTest, FullCoverageNoDuplicatesAllInBounds) {
  auto plan = computeGemmTilePlan(/*M=*/7, /*K=*/9, /*N=*/5, /*mu=*/3,
                                  /*minSlotCount=*/64);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(static_cast<int64_t>(plan->tasks.size()),
            plan->I * plan->J * plan->Q);

  std::set<std::tuple<int64_t, int64_t, int64_t>> seen;
  for (const GemmTask& task : plan->tasks) {
    EXPECT_GE(task.i, 0);
    EXPECT_LT(task.i, plan->I);
    EXPECT_GE(task.j, 0);
    EXPECT_LT(task.j, plan->J);
    EXPECT_GE(task.q, 0);
    EXPECT_LT(task.q, plan->Q);
    auto [it, inserted] = seen.insert({task.i, task.j, task.q});
    EXPECT_TRUE(inserted) << "duplicate task (" << task.i << "," << task.j
                          << "," << task.q << ")";
  }
  // Full coverage: every legal (i,j,q) triple was seen.
  for (int64_t i = 0; i < plan->I; ++i) {
    for (int64_t j = 0; j < plan->J; ++j) {
      for (int64_t q = 0; q < plan->Q; ++q) {
        EXPECT_TRUE(seen.count({i, j, q}) == 1)
            << "missing task (" << i << "," << j << "," << q << ")";
      }
    }
  }
  EXPECT_TRUE(succeeded(verifyGemmTilePlan(*plan)));
}

TEST(TilePlanningMtpJklsTest, CanonicalOrderIsOuterQInnerDestTileId) {
  auto plan = computeGemmTilePlan(/*M=*/5, /*K=*/6, /*N=*/3, /*mu=*/2,
                                  /*minSlotCount=*/64);
  ASSERT_TRUE(succeeded(plan));
  for (size_t idx = 0; idx < plan->tasks.size(); ++idx) {
    int64_t expectedQ = static_cast<int64_t>(idx) / plan->taskCount;
    int64_t expectedDestTileId = static_cast<int64_t>(idx) % plan->taskCount;
    EXPECT_EQ(plan->tasks[idx].q, expectedQ) << "at idx " << idx;
    EXPECT_EQ(plan->tasks[idx].destTileId, expectedDestTileId)
        << "at idx " << idx;
  }
}

TEST(TilePlanningMtpJklsTest, BoundaryExtentsMatchOracleEverywhere) {
  // A single moderately irregular shape exercising boundaries in all three
  // directions at once, checked against an independent oracle formula.
  int64_t M = 7, K = 5, N = 8, mu = 3;
  auto plan = computeGemmTilePlan(M, K, N, mu, /*minSlotCount=*/128);
  ASSERT_TRUE(succeeded(plan));
  for (const GemmTask& task : plan->tasks) {
    EXPECT_EQ(task.validRows, oracleValidExtent(M, task.i, mu));
    EXPECT_EQ(task.validContraction, oracleValidExtent(K, task.q, mu));
    EXPECT_EQ(task.validColumns, oracleValidExtent(N, task.j, mu));
  }
}

// ---------------------------------------------------------------------
// Reuse identities and reduction chain
// ---------------------------------------------------------------------

TEST(TilePlanningMtpJklsTest, LhsTileReusedAcrossJRhsTileReusedAcrossI) {
  auto plan = computeGemmTilePlan(/*M=*/4, /*K=*/4, /*N=*/4, /*mu=*/2,
                                  /*minSlotCount=*/32);
  ASSERT_TRUE(succeeded(plan));
  for (const GemmTask& a : plan->tasks) {
    for (const GemmTask& b : plan->tasks) {
      bool sameIQ = a.i == b.i && a.q == b.q;
      EXPECT_EQ(a.lhsTileId == b.lhsTileId, sameIQ);
      bool sameQJ = a.q == b.q && a.j == b.j;
      EXPECT_EQ(a.rhsTileId == b.rhsTileId, sameQJ);
      bool sameIJ = a.i == b.i && a.j == b.j;
      EXPECT_EQ(a.destTileId == b.destTileId, sameIJ);
    }
  }
}

TEST(TilePlanningMtpJklsTest, ReductionChainCoversEveryQInOrderForEveryDest) {
  auto plan = computeGemmTilePlan(/*M=*/5, /*K=*/7, /*N=*/3, /*mu=*/2,
                                  /*minSlotCount=*/64);
  ASSERT_TRUE(succeeded(plan));
  for (int64_t i = 0; i < plan->I; ++i) {
    for (int64_t j = 0; j < plan->J; ++j) {
      std::vector<GemmTask> chain = plan->reductionChainForDest(i, j);
      ASSERT_EQ(static_cast<int64_t>(chain.size()), plan->Q);
      for (int64_t q = 0; q < plan->Q; ++q) {
        EXPECT_EQ(chain[q].q, q);
        EXPECT_EQ(chain[q].i, i);
        EXPECT_EQ(chain[q].j, j);
      }
    }
  }
}

TEST(TilePlanningMtpJklsTest, ReductionChainOutOfBoundsIsEmpty) {
  auto plan = computeGemmTilePlan(/*M=*/4, /*K=*/4, /*N=*/4, /*mu=*/2,
                                  /*minSlotCount=*/32);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_TRUE(plan->reductionChainForDest(-1, 0).empty());
  EXPECT_TRUE(plan->reductionChainForDest(0, -1).empty());
  EXPECT_TRUE(plan->reductionChainForDest(plan->I, 0).empty());
  EXPECT_TRUE(plan->reductionChainForDest(0, plan->J).empty());
}

// ---------------------------------------------------------------------
// Physical group/position: must agree with getBalancedMtpPacking directly
// ---------------------------------------------------------------------

TEST(TilePlanningMtpJklsTest, GroupPositionMatchesBalancedPackingDirectly) {
  int64_t M = 6, K = 6, N = 6, mu = 2, minSlotCount = 8;  // taskCount=9
  auto plan = computeGemmTilePlan(M, K, N, mu, minSlotCount);
  ASSERT_TRUE(succeeded(plan));

  auto packing = getBalancedMtpPacking(plan->taskCount, mu, mu, minSlotCount);
  ASSERT_TRUE(succeeded(packing));
  EXPECT_EQ(plan->numCiphertexts, packing->numCiphertexts);
  EXPECT_EQ(plan->tilesPerCiphertext, packing->tilesPerCiphertext);

  for (const GemmTask& task : plan->tasks) {
    EXPECT_EQ(task.group, task.destTileId / packing->tilesPerCiphertext);
    EXPECT_EQ(task.position, task.destTileId % packing->tilesPerCiphertext);
    EXPECT_GE(task.group, 0);
    EXPECT_LT(task.group, plan->numCiphertexts);
    EXPECT_GE(task.position, 0);
    EXPECT_LT(task.position, plan->tilesPerCiphertext);
  }
}

TEST(TilePlanningMtpJklsTest, MultipleCiphertextGroupsWithUnusedPosition) {
  // taskCount=3, capacity=floor(8/4)=2: numCiphertexts=2,
  // tilesPerCiphertext=2 -- mirrors Phase 7's uneven-batch case, one
  // ciphertext has an unused tile position.
  auto plan = computeGemmTilePlan(/*M=*/2, /*K=*/2, /*N=*/6, /*mu=*/2,
                                  /*minSlotCount=*/8);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(plan->taskCount, 3);
  EXPECT_EQ(plan->numCiphertexts, 2);
  EXPECT_EQ(plan->tilesPerCiphertext, 2);

  std::set<std::pair<int64_t, int64_t>> occupied;
  for (const GemmTask& task : plan->tasks) {
    occupied.insert({task.group, task.position});
  }
  // 3 destination tiles occupy 3 of the 2*2=4 (group,position) slots; one
  // is legitimately unused (not occupied by any task).
  EXPECT_EQ(occupied.size(), 3u);
}

// ---------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------

TEST(TilePlanningMtpJklsTest, ConstructionIsDeterministic) {
  auto plan1 = computeGemmTilePlan(/*M=*/5, /*K=*/7, /*N=*/9, /*mu=*/3,
                                   /*minSlotCount=*/64);
  auto plan2 = computeGemmTilePlan(/*M=*/5, /*K=*/7, /*N=*/9, /*mu=*/3,
                                   /*minSlotCount=*/64);
  ASSERT_TRUE(succeeded(plan1));
  ASSERT_TRUE(succeeded(plan2));
  ASSERT_EQ(plan1->tasks.size(), plan2->tasks.size());
  for (size_t idx = 0; idx < plan1->tasks.size(); ++idx) {
    const GemmTask& a = plan1->tasks[idx];
    const GemmTask& b = plan2->tasks[idx];
    EXPECT_EQ(a.i, b.i);
    EXPECT_EQ(a.j, b.j);
    EXPECT_EQ(a.q, b.q);
    EXPECT_EQ(a.validRows, b.validRows);
    EXPECT_EQ(a.validContraction, b.validContraction);
    EXPECT_EQ(a.validColumns, b.validColumns);
    EXPECT_EQ(a.lhsTileId, b.lhsTileId);
    EXPECT_EQ(a.rhsTileId, b.rhsTileId);
    EXPECT_EQ(a.destTileId, b.destTileId);
    EXPECT_EQ(a.group, b.group);
    EXPECT_EQ(a.position, b.position);
  }
}

// ---------------------------------------------------------------------
// Invalid inputs: clean rejection, not a crash
// ---------------------------------------------------------------------

TEST(TilePlanningMtpJklsTest, RejectsNonPositiveDimensions) {
  EXPECT_TRUE(failed(computeGemmTilePlan(0, 2, 2, 2, 32)));
  EXPECT_TRUE(failed(computeGemmTilePlan(-1, 2, 2, 2, 32)));
  EXPECT_TRUE(failed(computeGemmTilePlan(2, 0, 2, 2, 32)));
  EXPECT_TRUE(failed(computeGemmTilePlan(2, -3, 2, 2, 32)));
  EXPECT_TRUE(failed(computeGemmTilePlan(2, 2, 0, 2, 32)));
  EXPECT_TRUE(failed(computeGemmTilePlan(2, 2, -4, 2, 32)));
  EXPECT_TRUE(failed(computeGemmTilePlan(2, 2, 2, 0, 32)));
  EXPECT_TRUE(failed(computeGemmTilePlan(2, 2, 2, -2, 32)));
  EXPECT_TRUE(failed(computeGemmTilePlan(2, 2, 2, 2, 0)));
  EXPECT_TRUE(failed(computeGemmTilePlan(2, 2, 2, 2, -1)));
}

TEST(TilePlanningMtpJklsTest, RejectsTileTooLargeForOneCiphertext) {
  // mu*mu=16 > minSlotCount=8: no single primitive tile fits at all.
  EXPECT_TRUE(failed(computeGemmTilePlan(4, 4, 4, 4, 8)));
}

TEST(TilePlanningMtpJklsTest, RejectsOverflowingTaskCount) {
  constexpr int64_t kHuge = std::numeric_limits<int64_t>::max() / 2;
  // I and J would each be enormous (M, N huge; mu=1), so I*J overflows.
  EXPECT_TRUE(failed(computeGemmTilePlan(kHuge, 2, kHuge, 1, 8)));
}

TEST(TilePlanningMtpJklsTest, RejectsOverflowingTotalTaskCount) {
  // taskCount itself is representable, but taskCount*Q is not.
  constexpr int64_t kHuge = std::numeric_limits<int64_t>::max() / 2;
  EXPECT_TRUE(failed(computeGemmTilePlan(4, kHuge, 4, 1, 8)));
}

TEST(TilePlanningMtpJklsTest, RejectsWhenNoTileFitsForHugeMu) {
  // mu itself so large that mu*mu would overflow before even comparing to
  // minSlotCount -- getBalancedMtpPacking must reject this safely, and so
  // must computeGemmTilePlan by propagating that failure.
  constexpr int64_t kHuge = std::numeric_limits<int64_t>::max() / 2;
  EXPECT_TRUE(failed(computeGemmTilePlan(4, 4, 4, kHuge, 8)));
}

// ---------------------------------------------------------------------
// verifyGemmTilePlan as an independent oracle: must also reject malformed
// plans a caller (or a future materializer) might construct incorrectly,
// not just accept whatever computeGemmTilePlan itself produces.
// ---------------------------------------------------------------------

TEST(TilePlanningMtpJklsTest, VerifyAcceptsEveryValidPlanConstructed) {
  struct Shape {
    int64_t M, K, N, mu, minSlotCount;
  };
  const Shape shapes[] = {
      {2, 2, 2, 2, 8},    {2, 2, 4, 2, 8},   {2, 4, 2, 2, 8},
      {3, 5, 4, 2, 32},   {4, 4, 3, 2, 16},  {7, 9, 5, 3, 64},
      {1, 1, 1, 1, 4},    {3, 3, 3, 4, 32},  {10, 10, 10, 3, 128},
  };
  for (const Shape& s : shapes) {
    auto plan = computeGemmTilePlan(s.M, s.K, s.N, s.mu, s.minSlotCount);
    ASSERT_TRUE(succeeded(plan)) << "M=" << s.M << " K=" << s.K
                                 << " N=" << s.N << " mu=" << s.mu;
    EXPECT_TRUE(succeeded(verifyGemmTilePlan(*plan)))
        << "M=" << s.M << " K=" << s.K << " N=" << s.N << " mu=" << s.mu;
  }
}

TEST(TilePlanningMtpJklsTest, VerifyRejectsWrongTaskCount) {
  auto plan = computeGemmTilePlan(4, 4, 4, 2, 32);
  ASSERT_TRUE(succeeded(plan));
  GemmTilePlan bad = *plan;
  bad.tasks.pop_back();  // now short of I*J*Q
  EXPECT_TRUE(failed(verifyGemmTilePlan(bad)));
}

TEST(TilePlanningMtpJklsTest, VerifyRejectsDuplicateTaskOverwritingAnother) {
  auto plan = computeGemmTilePlan(4, 4, 4, 2, 32);
  ASSERT_TRUE(succeeded(plan));
  GemmTilePlan bad = *plan;
  // Overwrite the second task with a duplicate of the first, without
  // changing the vector's size -- must be caught by the per-position
  // canonical-order check, not silently accepted.
  bad.tasks[1] = bad.tasks[0];
  EXPECT_TRUE(failed(verifyGemmTilePlan(bad)));
}

TEST(TilePlanningMtpJklsTest, VerifyRejectsOutOfBoundsCoordinate) {
  auto plan = computeGemmTilePlan(4, 4, 4, 2, 32);
  ASSERT_TRUE(succeeded(plan));
  GemmTilePlan bad = *plan;
  bad.tasks[0].i = bad.I;  // one past the end
  EXPECT_TRUE(failed(verifyGemmTilePlan(bad)));
}

TEST(TilePlanningMtpJklsTest, VerifyRejectsWrongBoundaryExtent) {
  auto plan = computeGemmTilePlan(3, 5, 4, 2, 32);  // has real boundaries
  ASSERT_TRUE(succeeded(plan));
  GemmTilePlan bad = *plan;
  bad.tasks[0].validRows = bad.mu + 1;  // impossible: exceeds mu
  EXPECT_TRUE(failed(verifyGemmTilePlan(bad)));
}

TEST(TilePlanningMtpJklsTest, VerifyRejectsWrongGroupPosition) {
  auto plan = computeGemmTilePlan(2, 2, 6, 2, 8);  // multi-ciphertext
  ASSERT_TRUE(succeeded(plan));
  GemmTilePlan bad = *plan;
  bad.tasks[0].group = bad.numCiphertexts;  // one past the end
  EXPECT_TRUE(failed(verifyGemmTilePlan(bad)));
}

TEST(TilePlanningMtpJklsTest, VerifyRejectsPlanWithBadTopLevelFields) {
  auto plan = computeGemmTilePlan(4, 4, 4, 2, 32);
  ASSERT_TRUE(succeeded(plan));
  {
    GemmTilePlan bad = *plan;
    bad.I = bad.I + 1;  // disagrees with ceil(M/mu)
    EXPECT_TRUE(failed(verifyGemmTilePlan(bad)));
  }
  {
    GemmTilePlan bad = *plan;
    bad.taskCount = bad.taskCount + 1;  // disagrees with I*J
    EXPECT_TRUE(failed(verifyGemmTilePlan(bad)));
  }
  {
    GemmTilePlan bad = *plan;
    bad.tilesPerCiphertext = bad.tilesPerCiphertext + 1;  // disagrees with getBalancedMtpPacking
    EXPECT_TRUE(failed(verifyGemmTilePlan(bad)));
  }
  {
    GemmTilePlan bad = *plan;
    bad.mu = 0;  // non-positive
    EXPECT_TRUE(failed(verifyGemmTilePlan(bad)));
  }
}

}  // namespace
}  // namespace heir
}  // namespace mlir
