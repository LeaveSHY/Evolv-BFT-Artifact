package hotstuff

import "testing"

func TestRankState_ExpectedAndVerify(t *testing.T) {
	rs := NewRankState(1, 4)
	if rs.ExpectedRank(0) != 1 {
		t.Fatalf("unexpected rank for height 0: %d", rs.ExpectedRank(0))
	}
	if rs.ExpectedRank(2) != 9 {
		t.Fatalf("unexpected rank for height 2: %d", rs.ExpectedRank(2))
	}
	if !rs.VerifyRank(2, 9) {
		t.Fatalf("expected verify true")
	}
	if rs.VerifyRank(2, 8) {
		t.Fatalf("expected verify false")
	}
}

