#include "TROOT.h"
#include "TKey.h"
#include "TFile.h"
#include "TSystem.h"
#include "TCanvas.h"
#include "TTree.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TObjArray.h"
#include "THStack.h"
#include "TLegend.h"
#include "TEfficiency.h"
#include "TGraphAsymmErrors.h"
#include "TF1.h"
#include "TMath.h"
#include "TCut.h"
#include "TPaletteAxis.h"
#include "TLatex.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string>
#include <algorithm>

void plotAnyVariable(string varName, string category, string label, int nbins, float min, float max, bool RT, bool logy){

  
  //TFile *File = TFile::Open("/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_Training_SA_31082019/09092019_Newlytrained/Ntuple/out_singlephoton_ntuple_2018Gjet_SAPresel_Mgg95_09092019_uncorr_Train.root");

  TFile *File = TFile::Open("Ntuple/Output_SinglePhoton_GJet_UL16_PreVFP_PostVFP_photonIscorrectionEE_Added_Train_20042022.root");

  TTree *t_FakePhotons = (TTree*)File->Get("fakePhotons");
  TTree *t_PromtPhotons = (TTree*)File->Get("promptPhotons");

  TH1F * histo_fake = new TH1F ("histo_fake","histo_fake",nbins,min,max);
  TH1F * histo_promt = new TH1F ("histo_promt","histo_promt",nbins,min,max);
  TH1F * histo_promt_with2Dw = new TH1F ("histo_promt_with2Dw","histo_promt_with2Dw",nbins,min,max);

  TString tmp_fake = "";
  tmp_fake += varName;
  tmp_fake+=">>histo_fake";

  TString tmp_promt = "";
  tmp_promt += varName;
  tmp_promt +=">>histo_promt";

  TString tmp_promt_2Dw = "";
  tmp_promt_2Dw += varName;
  tmp_promt_2Dw +=">>histo_promt_with2Dw";

  if(category == "eb") {
    t_FakePhotons->Draw(tmp_fake,"(abs(scEta)<1.4442 && pt > 18)*weight","goff");
    t_PromtPhotons->Draw(tmp_promt,"(abs(scEta)<1.4442 && pt > 18)*weight","goff");
    t_PromtPhotons->Draw(tmp_promt_2Dw,"(abs(scEta)<1.4442 && pt > 18)*weight*PtvsEtaWeight","goff");
  } else if(category == "ee") {
    t_FakePhotons->Draw(tmp_fake,"(abs(scEta)>1.566 && pt > 18)*weight","goff");
    t_PromtPhotons->Draw(tmp_promt,"(abs(scEta)>1.566 && pt > 18)*weight","goff");
    t_PromtPhotons->Draw(tmp_promt_2Dw,"(abs(scEta)>1.566 && pt > 18)*weight*PtvsEtaWeight","goff");
  } else if(category == "all") {
    t_FakePhotons->Draw(tmp_fake,"(pt > 18)*weight","goff");
    t_PromtPhotons->Draw(tmp_promt,"(pt > 18)*weight","goff");
    t_PromtPhotons->Draw(tmp_promt_2Dw,"(pt > 18)*weight*PtvsEtaWeight","goff");
  }


  histo_fake->Sumw2();
  histo_promt->Sumw2();
  histo_promt_with2Dw->Sumw2();

  histo_fake->Scale(histo_promt->Integral()/histo_fake->Integral());
  histo_promt_with2Dw->Scale(histo_promt->Integral()/histo_promt_with2Dw->Integral());

  TCanvas * can = new TCanvas("can","can",600,600);
  can->cd();
  if(logy) can->SetLogy();

  histo_fake->SetTitle("");
  histo_fake->SetStats(0);
  histo_fake->GetXaxis()->SetTitle(label.c_str());
  histo_fake->GetXaxis()->SetLabelSize(0.03);
  histo_fake->GetYaxis()->SetTitle("Events");
  // histo_fake->GetYaxis()->SetTitleOffset();
  histo_fake->GetYaxis()->SetLabelSize(0.03);

  if(!logy) {
    histo_fake->Scale(0.001);
    histo_promt->Scale(0.001);
    histo_promt_with2Dw->Scale(0.001);
  }

  histo_fake->SetMinimum(0.);
  histo_promt->SetMinimum(0.);

  if(logy) {
  histo_fake->SetMinimum(10.);
  histo_promt->SetMinimum(10.);
  }

  double maxVal_prompt = histo_promt->GetBinContent(histo_promt->GetMaximumBin());
  double maxVal_fake = histo_fake->GetBinContent(histo_fake->GetMaximumBin());
  double maxVal_prompt_with2Dw = histo_promt_with2Dw->GetBinContent(histo_promt_with2Dw->GetMaximumBin());
  double maxVal = std::max(std::max(maxVal_prompt,maxVal_fake),std::max(maxVal_prompt,maxVal_prompt_with2Dw));

  double maxValue = maxVal*1.2;

  /* if(maxVal_prompt > maxVal_fake){
    maxValue = maxVal_prompt*1.2;
  }
  else{
    maxValue = maxVal_fake*1.2;
    }*/

  histo_fake->SetLineColor(kRed);
  histo_fake->SetLineWidth(2);
  histo_fake->SetMaximum(maxValue);
  histo_fake->Draw("HIST");
  
  histo_promt->SetLineColor(kBlue);
  histo_promt->SetLineWidth(2);
  histo_promt->Draw("HISTSAME");

  histo_promt_with2Dw->SetLineColor(kGreen+1);
  histo_promt_with2Dw->SetLineWidth(2);
  histo_promt_with2Dw->Draw("HISTSAME");

  float x1=0.15, x2=0.45;
  if(RT) x1=0.60, x2=0.90;

  TLegend *legend = new TLegend(x1, 0.6, x2, 0.9, "", "brNDC");
  legend->SetHeader("");
  if(category == "eb") legend->SetHeader("EB");
  else if(category == "ee") legend->SetHeader("EE");

  legend->AddEntry(histo_fake,"bkg","l");
  legend->AddEntry(histo_promt,"sig","l");
  legend->AddEntry(histo_promt_with2Dw,"sig * 2D weight","l");

  legend->SetBorderSize(0);
  legend->SetFillStyle(0);
  legend->SetTextFont(42);

  legend->Draw("same");

  TLatex *txt = new TLatex(0.2, 0.9, "");
  // txt->SetTextSize(0.05);
  txt->DrawLatexNDC(0.1, 0.91, "CMS #bf{#it{#scale[0.8]{Simulation Preliminary}}}");
  txt->DrawLatexNDC(0.76, 0.91, "#bf{13 TeV}");
  if(!logy) txt->DrawLatexNDC(0.04, 0.88, "#scale[0.6]{#bf{#times10^{3}}}");
  txt->Draw("same");
  
  can->Update();
  can->Modified();

  can->SaveAs((varName+"_"+category+"_UL16MC_SA_Hgg.pdf").c_str());
  can->SaveAs((varName+"_"+category+"_UL16MC_SA_Hgg.png").c_str());
  can->SaveAs((varName+"_"+category+"_UL16MC_SA_Hgg.root").c_str());
}

void plotinput_vars(){
  // plotAnyVariable(string varName, string category, string label, int nbins, float min, float max, bool RT, bool logy)

  //plotAnyVariable("isoPhoCorrMax2p5", "ee", "isoPhoCorrMax2p5", 100, -5.0, 10., true, true);
  plotAnyVariable("phoIso03", "ee", "PF Photon ISO (cone 03)", 100, -0.0005, 0.0005, true, true);

  /*plotAnyVariable("r9", "eb", "R9", 70, 0.4, 1.2, false, false);
  plotAnyVariable("r9", "ee", "R9", 70, 0.4, 1.2, false, false);
  plotAnyVariable("s4", "eb", "E_{2#times2}/E_{5#times5}", 90, 0.2, 1.1, false, false);
  plotAnyVariable("s4", "ee", "E_{2#times2}/E_{5#times5}", 90, 0.2, 1.1, false, false);
  plotAnyVariable("sigmaIetaIeta", "eb", "#sigma_{i#etai#eta}", 50, 0.0, 0.025, true, false);
  plotAnyVariable("sigmaIetaIeta", "ee", "#sigma_{i#etai#eta}", 60, 0.0, 0.06, true, false);
  plotAnyVariable("covIEtaIPhi", "eb", "COV_{i#etai#phi}", 60, -0.0003, 0.0003, false, false);
  plotAnyVariable("covIEtaIPhi", "ee", "COV_{i#etai#phi}", 100, -0.001, 0.001, false, false);
  plotAnyVariable("etaWidth", "eb", "#sigma_{#eta}", 50, 0.0, 0.025, true, false);
  plotAnyVariable("etaWidth", "ee", "#sigma_{#eta}", 60, 0.0, 0.06, true, false);
  plotAnyVariable("phiWidth", "eb", "#sigma_{#phi}", 50, 0.0, 0.1, true, false);
  plotAnyVariable("phiWidth", "ee", "#sigma_{#phi}", 50, 0.0, 0.1, true, false);
  plotAnyVariable("esEffSigmaRR", "ee", "ES #sigma_{effRR}", 50, 0.0, 15.0, true, true);
  plotAnyVariable("esEnergyOverRawE", "ee", "E_{ES}/Raw E_{SC}", 100, 0.0, 0.2, true, true);
  plotAnyVariable("SCRawE", "eb", "Supercluster E_{RAW}", 100, 0.0, 400., true, false);
  plotAnyVariable("SCRawE", "ee", "Supercluster E_{RAW}", 100, 0.0, 800., true, false);
  plotAnyVariable("phoIso03", "eb", "PF Photon ISO (cone 03)", 100, 0.0, 10., true, true);
  plotAnyVariable("phoIso03", "ee", "PF Photon ISO (cone 03)", 100, 0.0, 10., true, true);
  plotAnyVariable("isoPhoCorrMax2p5", "ee", "isoPhoCorrMax2p5", 100, 0.0, 10., true, true);
  plotAnyVariable("chgIsoWrtChosenVtx", "eb", "PF Charged ISO (chosen vertex)", 100, 0.0, 10., true, true);
  plotAnyVariable("chgIsoWrtChosenVtx", "ee", "PF Charged ISO (chosen vertex)", 100, 0.0, 10., true, true);
  plotAnyVariable("chgIsoWrtWorstVtx", "eb", "PF Charged ISO (worst vertex)", 100, 0.0, 10., true, true);
  plotAnyVariable("chgIsoWrtWorstVtx", "ee", "PF Charged ISO (worst vertex)", 100, 0.0, 10., true, true);
  plotAnyVariable("rho", "eb", "#rho", 50, 0.0, 50.0, true, false);
  plotAnyVariable("rho", "ee", "#rho", 50, 0.0, 50.0, true, false); */
}
