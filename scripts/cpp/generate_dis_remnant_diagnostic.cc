#include "Pythia8/Pythia.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace Pythia8;

namespace {

struct BeamEntry {
  int i = -1;
  int iPos = -1;
  int id = 0;
  double x = 0.0;
  double px = 0.0;
  double py = 0.0;
  double pz = 0.0;
  double E = 0.0;
  bool isValence = false;
  bool isCompanion = false;
  bool isFromBeam = false;
  std::string category;
};

struct CsvRow {
  int event_id = 0;
  int struck_incoming_index = -1;
  int struck_incoming_pdg_id = 0;
  std::string struck_incoming_px, struck_incoming_py, struck_incoming_pz, struck_incoming_E;
  int beamB_n_entries = 0;
  int beamB_sizeInit = 0;
  int n_elementary_quark_entries = 0;
  int n_elementary_quark_entries_excluding_initiator = 0;
  int n_valence_quark_candidates_excluding_initiator = 0;
  int n_companion_entries = 0;
  int n_diquark_like_entries = 0;
  int n_baryon_or_composite_entries = 0;
  std::string remnant_classification;
  std::string candidate_q1_beam_index, candidate_q1_pdg_id, candidate_q1_x, candidate_q1_px, candidate_q1_py, candidate_q1_pz, candidate_q1_E, candidate_q1_isValence, candidate_q1_isCompanion;
  std::string candidate_q2_beam_index, candidate_q2_pdg_id, candidate_q2_x, candidate_q2_px, candidate_q2_py, candidate_q2_pz, candidate_q2_E, candidate_q2_isValence, candidate_q2_isCompanion;
  std::string likely_initiator_beam_index, likely_initiator_x, likely_initiator_iPos;
  int final_remnant_record_index = -1;
  int final_remnant_record_pdg_id = 0;
  std::string final_remnant_record_px, final_remnant_record_py, final_remnant_record_pz, final_remnant_record_E;
  std::string notes;
};

std::string fmt(double x) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(6) << x;
  return os.str();
}

void logln(std::ostream& log, const std::string& s) {
  std::cout << s << "\n";
  log << s << "\n";
}

int pickIncomingQuarkIndex(const Event& ev) {
  int best = -1;
  double bestAbsPz = -1.0;
  for (int i = 0; i < ev.size(); ++i) {
    const Particle& p = ev[i];
    if (p.status() >= 0) continue;
    if (std::abs(p.id()) > 6) continue;
    double apz = std::abs(p.pz());
    if (apz > bestAbsPz) {
      bestAbsPz = apz;
      best = i;
    }
  }
  return best;
}

bool isElementaryQuark(int id) { return std::abs(id) >= 1 && std::abs(id) <= 6; }
bool isGluon(int id) { return id == 21; }
bool isDiquarkLike(int id) { return std::abs(id) >= 1000 && std::abs(id) < 10000; }
bool isBaryonCompositeLike(int id) { return std::abs(id) >= 100 && !isElementaryQuark(id) && !isGluon(id); }

std::string categoryFromId(int id) {
  if (isElementaryQuark(id)) return "elementary_quark";
  if (isGluon(id)) return "gluon";
  if (isDiquarkLike(id)) return "diquark_like";
  if (isBaryonCompositeLike(id)) return "baryon_or_composite";
  return "other";
}

void writeCsvHeader(std::ofstream& csv) {
  csv
      << "event_id,struck_incoming_index,struck_incoming_pdg_id,"
      << "struck_incoming_px,struck_incoming_py,struck_incoming_pz,struck_incoming_E,"
      << "beamB_n_entries,beamB_sizeInit,"
      << "n_elementary_quark_entries,n_elementary_quark_entries_excluding_initiator,"
      << "n_valence_quark_candidates_excluding_initiator,n_companion_entries,"
      << "n_diquark_like_entries,n_baryon_or_composite_entries,"
      << "remnant_classification,"
      << "candidate_q1_beam_index,candidate_q1_pdg_id,candidate_q1_x,candidate_q1_px,candidate_q1_py,candidate_q1_pz,candidate_q1_E,candidate_q1_isValence,candidate_q1_isCompanion,"
      << "candidate_q2_beam_index,candidate_q2_pdg_id,candidate_q2_x,candidate_q2_px,candidate_q2_py,candidate_q2_pz,candidate_q2_E,candidate_q2_isValence,candidate_q2_isCompanion,"
      << "likely_initiator_beam_index,likely_initiator_x,likely_initiator_iPos,"
      << "final_remnant_record_index,final_remnant_record_pdg_id,final_remnant_record_px,final_remnant_record_py,final_remnant_record_pz,final_remnant_record_E,notes\n";
}

std::string q(const std::string& s) {
  std::string out = "\"";
  for (char c : s) out += (c == '"') ? "'" : std::string(1, c);
  out += "\"";
  return out;
}

void writeCsvRow(std::ofstream& csv, const CsvRow& r) {
  csv << r.event_id << "," << r.struck_incoming_index << "," << r.struck_incoming_pdg_id << ","
      << q(r.struck_incoming_px) << "," << q(r.struck_incoming_py) << "," << q(r.struck_incoming_pz) << "," << q(r.struck_incoming_E) << ","
      << r.beamB_n_entries << "," << r.beamB_sizeInit << ","
      << r.n_elementary_quark_entries << "," << r.n_elementary_quark_entries_excluding_initiator << ","
      << r.n_valence_quark_candidates_excluding_initiator << "," << r.n_companion_entries << ","
      << r.n_diquark_like_entries << "," << r.n_baryon_or_composite_entries << ","
      << q(r.remnant_classification) << ","
      << q(r.candidate_q1_beam_index) << "," << q(r.candidate_q1_pdg_id) << "," << q(r.candidate_q1_x) << ","
      << q(r.candidate_q1_px) << "," << q(r.candidate_q1_py) << "," << q(r.candidate_q1_pz) << "," << q(r.candidate_q1_E) << ","
      << q(r.candidate_q1_isValence) << "," << q(r.candidate_q1_isCompanion) << ","
      << q(r.candidate_q2_beam_index) << "," << q(r.candidate_q2_pdg_id) << "," << q(r.candidate_q2_x) << ","
      << q(r.candidate_q2_px) << "," << q(r.candidate_q2_py) << "," << q(r.candidate_q2_pz) << "," << q(r.candidate_q2_E) << ","
      << q(r.candidate_q2_isValence) << "," << q(r.candidate_q2_isCompanion) << ","
      << q(r.likely_initiator_beam_index) << "," << q(r.likely_initiator_x) << "," << q(r.likely_initiator_iPos) << ","
      << r.final_remnant_record_index << "," << r.final_remnant_record_pdg_id << ","
      << q(r.final_remnant_record_px) << "," << q(r.final_remnant_record_py) << ","
      << q(r.final_remnant_record_pz) << "," << q(r.final_remnant_record_E) << ","
      << q(r.notes) << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  int targetAccepted = 5;
  if (argc > 1) targetAccepted = std::max(1, std::atoi(argv[1]));

  std::filesystem::path outdir =
      "/Users/johnterry/Documents/Projects/Di-quark-pythia/outputs/dis_isr_parton_dataset_cpp";
  std::filesystem::create_directories(outdir);
  std::filesystem::path logPath = outdir / "remnant_cpp_diagnostic_5events.txt";
  std::filesystem::path csvPath = outdir / "remnant_constituents_diagnostic.csv";
  std::ofstream log(logPath);
  std::ofstream csv(csvPath);
  writeCsvHeader(csv);

  Pythia pythia;
  pythia.readString("Beams:idA = 11");
  pythia.readString("Beams:idB = 2212");
  pythia.readString("Beams:eA = 18.0");
  pythia.readString("Beams:eB = 275.0");
  pythia.readString("Beams:frameType = 2");
  pythia.readString("WeakBosonExchange:ff2ff(t:gmZ) = on");
  pythia.readString("PhaseSpace:Q2Min = 16.0");
  pythia.readString("ProcessLevel:all = on");
  pythia.readString("PartonLevel:all = on");
  pythia.readString("PDF:lepton = off");
  pythia.readString("PartonLevel:ISR = off");
  pythia.readString("PartonLevel:FSR = off");
  pythia.readString("PartonLevel:MPI = off");
  pythia.readString("PartonLevel:Remnants = on");
  pythia.readString("HadronLevel:all = off");
  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = 12345");
  pythia.readString("Print:quiet = on");
  pythia.init();

  logln(log, "=== C++ remnant introspection ===");
  logln(log, "Category definitions:");
  logln(log, "  elementary_quark: abs(id) in {1,2,3,4,5,6}");
  logln(log, "  gluon: id == 21");
  logln(log, "  diquark_like: 1000 <= abs(id) < 10000");
  logln(log, "  baryon_or_composite: abs(id) >= 100 and not elementary_quark/gluon");
  logln(log, "  other: everything else");
  logln(log, "Access to pythia.beamA / pythia.beamB compiles in C++.");
  logln(log, "beamA.id=" + std::to_string(pythia.beamA.id()) +
                  " beamB.id=" + std::to_string(pythia.beamB.id()));
  logln(log, "beamB.isHadron=" + std::to_string(pythia.beamB.isHadron()) +
                  " isBaryon=" + std::to_string(pythia.beamB.isBaryon()) +
                  " isUnresolved=" + std::to_string(pythia.beamB.isUnresolved()));
  logln(log, "beamB.size()=" + std::to_string(pythia.beamB.size()) +
                  " beamB.sizeInit()=" + std::to_string(pythia.beamB.sizeInit()));
  logln(log, "Calling beamB.list() once:");
  pythia.beamB.list();
  logln(log, "=== Begin accepted-event diagnostics ===");

  int accepted = 0;
  int generated = 0;
  while (accepted < targetAccepted) {
    if (!pythia.next()) continue;
    ++generated;
    const Event& ev = pythia.event;
    int struckIdx = pickIncomingQuarkIndex(ev);
    if (struckIdx < 0) continue;
    if (std::abs(ev[struckIdx].id()) != 2) continue;

    ++accepted;
    CsvRow row;
    row.event_id = accepted;
    row.struck_incoming_index = struckIdx;
    row.struck_incoming_pdg_id = ev[struckIdx].id();
    row.struck_incoming_px = fmt(ev[struckIdx].px());
    row.struck_incoming_py = fmt(ev[struckIdx].py());
    row.struck_incoming_pz = fmt(ev[struckIdx].pz());
    row.struck_incoming_E = fmt(ev[struckIdx].e());

    logln(log, "");
    logln(log, "--- accepted_event=" + std::to_string(accepted) + " generated_seen=" + std::to_string(generated) + " ---");
    const Particle& struck = ev[struckIdx];
    logln(log, "struck incoming idx=" + std::to_string(struckIdx) + " id=" + std::to_string(struck.id()) +
                   " status=" + std::to_string(struck.status()) +
                   " p=(px,py,pz,E)=(" + fmt(struck.px()) + "," + fmt(struck.py()) + "," + fmt(struck.pz()) + "," + fmt(struck.e()) + ")");

    int protonIdx = -1;
    for (int i = 0; i < ev.size(); ++i) {
      if (ev[i].id() == 2212 && ev[i].status() < 0) {
        protonIdx = i;
        break;
      }
    }
    if (protonIdx >= 0) {
      const Particle& p = ev[protonIdx];
      logln(log, "proton record idx=" + std::to_string(protonIdx) + " status=" + std::to_string(p.status()) +
                     " p=(" + fmt(p.px()) + "," + fmt(p.py()) + "," + fmt(p.pz()) + "," + fmt(p.e()) + ")");
    } else {
      logln(log, "proton record entry not found");
    }

    // Final remnant-like objects in event record.
    int finalRemnantIdx = -1;
    for (int i = 0; i < ev.size(); ++i) {
      bool remLike = (ev[i].status() == 63 || std::abs(ev[i].id()) >= 1000);
      if (!remLike) continue;
      logln(log, "event remnant-like idx=" + std::to_string(i) + " id=" + std::to_string(ev[i].id()) +
                     " status=" + std::to_string(ev[i].status()) +
                     " mothers=(" + std::to_string(ev[i].mother1()) + "," + std::to_string(ev[i].mother2()) + ")" +
                     " p=(" + fmt(ev[i].px()) + "," + fmt(ev[i].py()) + "," + fmt(ev[i].pz()) + "," + fmt(ev[i].e()) + ")");
      if (finalRemnantIdx < 0 && ev[i].status() == 63) finalRemnantIdx = i;
    }
    if (finalRemnantIdx >= 0) {
      row.final_remnant_record_index = finalRemnantIdx;
      row.final_remnant_record_pdg_id = ev[finalRemnantIdx].id();
      row.final_remnant_record_px = fmt(ev[finalRemnantIdx].px());
      row.final_remnant_record_py = fmt(ev[finalRemnantIdx].py());
      row.final_remnant_record_pz = fmt(ev[finalRemnantIdx].pz());
      row.final_remnant_record_E = fmt(ev[finalRemnantIdx].e());
    }

    // BeamB resolved-parton/remnant internal dump.
    const BeamParticle& b = pythia.beamB;
    int nRes = b.size();
    int nInit = b.sizeInit();
    row.beamB_n_entries = nRes;
    row.beamB_sizeInit = nInit;
    logln(log, "beamB internals: size=" + std::to_string(nRes) + " sizeInit=" + std::to_string(nInit));
    logln(log, "Calling beamB.list() for this event:");
    b.list();

    std::vector<BeamEntry> entries;
    for (int i = 0; i < nRes; ++i) {
      const ResolvedParton& rp = b[i];
      BeamEntry be;
      be.i = i;
      be.iPos = rp.iPos();
      be.id = rp.id();
      be.x = rp.x();
      be.px = rp.px();
      be.py = rp.py();
      be.pz = rp.pz();
      be.E = rp.e();
      be.isValence = rp.isValence();
      be.isCompanion = rp.isCompanion();
      be.isFromBeam = rp.isFromBeam();
      be.category = categoryFromId(be.id);
      entries.push_back(be);

      std::ostringstream line;
      line << "  beamB[" << i << "]"
           << " iPos=" << be.iPos
           << " id=" << be.id
           << " category=" << be.category
           << " x=" << fmt(be.x)
           << " comp=" << rp.companion()
           << " flags(val=" << be.isValence << ",sea_unmatched=" << rp.isUnmatched()
           << ",companion=" << be.isCompanion << ",fromBeam=" << be.isFromBeam << ")"
           << " p=(" << fmt(be.px) << "," << fmt(be.py) << "," << fmt(be.pz) << "," << fmt(be.E) << ")";
      logln(log, line.str());
    }
    // classify counts
    for (const auto& be : entries) {
      if (be.category == "elementary_quark") row.n_elementary_quark_entries++;
      if (be.isCompanion) row.n_companion_entries++;
      if (be.category == "diquark_like") row.n_diquark_like_entries++;
      if (be.category == "baryon_or_composite") row.n_baryon_or_composite_entries++;
    }

    // likely initiator: elementary quark with closest iPos to struck index, then |x|.
    int likelyInit = -1;
    int bestDPos = 1e9;
    double bestAbsX = 1e18;
    for (const auto& be : entries) {
      if (!isElementaryQuark(be.id)) continue;
      int dPos = std::abs(be.iPos - struckIdx);
      double ax = std::abs(be.x);
      if (dPos < bestDPos || (dPos == bestDPos && ax < bestAbsX)) {
        likelyInit = be.i;
        bestDPos = dPos;
        bestAbsX = ax;
      }
    }
    if (likelyInit >= 0) {
      const auto& binit = entries[likelyInit];
      row.likely_initiator_beam_index = std::to_string(binit.i);
      row.likely_initiator_x = fmt(binit.x);
      row.likely_initiator_iPos = std::to_string(binit.iPos);
      logln(log, "likely initiator beamB entry: i=" + std::to_string(binit.i) +
                     " (reason: elementary quark with minimal |iPos - struck_index|)");
    } else {
      logln(log, "likely initiator beamB entry: not found");
    }

    // leftover elementary quark candidates excluding likely initiator
    std::vector<BeamEntry> leftoverElem;
    std::vector<BeamEntry> leftoverVal;
    for (const auto& be : entries) {
      if (be.i == likelyInit) continue;
      if (!isElementaryQuark(be.id)) continue;
      leftoverElem.push_back(be);
      if (be.isValence && !be.isCompanion) leftoverVal.push_back(be);
    }
    row.n_elementary_quark_entries_excluding_initiator = static_cast<int>(leftoverElem.size());
    row.n_valence_quark_candidates_excluding_initiator = static_cast<int>(leftoverVal.size());

    logln(log, "leftover elementary quark entries excluding initiator: " + std::to_string(leftoverElem.size()));
    for (const auto& be : leftoverElem) {
      logln(log, "  leftover q: i=" + std::to_string(be.i) + " id=" + std::to_string(be.id) +
                     " x=" + fmt(be.x) + " valence=" + std::to_string(be.isValence) +
                     " companion=" + std::to_string(be.isCompanion));
    }

    // choose two best candidate remnant quarks by valence non-companion first, then x descending.
    auto sortByXDesc = [](const BeamEntry& a, const BeamEntry& b) { return a.x > b.x; };
    std::sort(leftoverVal.begin(), leftoverVal.end(), sortByXDesc);
    std::sort(leftoverElem.begin(), leftoverElem.end(), sortByXDesc);

    auto fillCandidate = [](const BeamEntry& be, bool q1, CsvRow& row) {
      if (q1) {
        row.candidate_q1_beam_index = std::to_string(be.i);
        row.candidate_q1_pdg_id = std::to_string(be.id);
        row.candidate_q1_x = fmt(be.x);
        row.candidate_q1_px = fmt(be.px);
        row.candidate_q1_py = fmt(be.py);
        row.candidate_q1_pz = fmt(be.pz);
        row.candidate_q1_E = fmt(be.E);
        row.candidate_q1_isValence = be.isValence ? "1" : "0";
        row.candidate_q1_isCompanion = be.isCompanion ? "1" : "0";
      } else {
        row.candidate_q2_beam_index = std::to_string(be.i);
        row.candidate_q2_pdg_id = std::to_string(be.id);
        row.candidate_q2_x = fmt(be.x);
        row.candidate_q2_px = fmt(be.px);
        row.candidate_q2_py = fmt(be.py);
        row.candidate_q2_pz = fmt(be.pz);
        row.candidate_q2_E = fmt(be.E);
        row.candidate_q2_isValence = be.isValence ? "1" : "0";
        row.candidate_q2_isCompanion = be.isCompanion ? "1" : "0";
      }
    };

    if (leftoverVal.size() >= 2) {
      row.remnant_classification = "two_explicit_elementary_remnant_quarks";
      fillCandidate(leftoverVal[0], true, row);
      fillCandidate(leftoverVal[1], false, row);
      row.notes = "Two leftover valence non-companion elementary quarks found";
      logln(log, "event classification: two_explicit_elementary_remnant_quarks");
      logln(log, "tentative partner rule: choose highest-x leftover valence quark (q1)");
    } else if (leftoverElem.size() >= 1 && (row.n_diquark_like_entries + row.n_baryon_or_composite_entries) > 0) {
      row.remnant_classification = "one_explicit_elementary_remnant_quark_plus_composite";
      fillCandidate(leftoverElem[0], true, row);
      row.notes = "One leftover elementary quark plus composite remnant entries";
      logln(log, "event classification: one_explicit_elementary_remnant_quark_plus_composite");
    } else if ((row.n_diquark_like_entries + row.n_baryon_or_composite_entries) > 0) {
      row.remnant_classification = "composite_only";
      row.notes = "No clear leftover elementary remnant quarks; composite entries dominate";
      logln(log, "event classification: composite_only");
    } else {
      row.remnant_classification = "ambiguous";
      row.notes = "Unable to isolate remnant pattern";
      logln(log, "event classification: ambiguous");
    }

    writeCsvRow(csv, row);
  }

  logln(log, "");
  logln(log, "=== Final summary ===");
  logln(log, "generated=" + std::to_string(generated) + " accepted=" + std::to_string(accepted));
  logln(log, "log_path=" + logPath.string());
  logln(log, "csv_path=" + csvPath.string());
  return 0;
}

