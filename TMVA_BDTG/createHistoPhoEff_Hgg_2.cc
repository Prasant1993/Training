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

#include "TMVA/Reader.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string>
#include <algorithm>


void saveHistoToFile(){

  string fileName_EB = "mvares_1.root";

  string fileName_EE = "mvares_2.root";

  // Reading tree for EB
  TFile *file_EB = new TFile(fileName_EB.c_str());
  TTree *t_EB_s = (TTree*)file_EB->Get("promptPhotons");
  TTree *t_EB_b = (TTree*)file_EB->Get("fakePhotons");

  // EB, no mvares cut
  TH1F * h_pt_EB_s    = new TH1F ("h_pt_EB_s","h_pt_EB_s",25,0,250);
  TH1F * h_pt_EB_b    = new TH1F ("h_pt_EB_b","h_pt_EB_b",25,0,250);
  TH1F * h_nVtx_EB_s  = new TH1F ("h_nVtx_EB_s","h_nVtx_EB_s",80,0,80);
  TH1F * h_nVtx_EB_b  = new TH1F ("h_nVtx_EB_b","h_nVtx_EB_b",80,0,80);
  TH1F * h_eta_EB_s   = new TH1F ("h_eta_EB_s","h_eta_EB_s",50,-2.5,2.5);
  TH1F * h_eta_EB_b   = new TH1F ("h_eta_EB_b","h_eta_EB_b",50,-2.5,2.5);
  TH1F * h_rho_EB_s   = new TH1F ("h_rho_EB_s","h_rho_EB_s",50,0,50);
  TH1F * h_rho_EB_b   = new TH1F ("h_rho_EB_b","h_rho_EB_b",50,0,50);

  TCut EB_s = "(mvares>-1 && abs(scEta)<1.5 && pt > 18)";
  TCut EB_b = "(mvares>-1 && abs(scEta)<1.5 && pt > 18)";

  h_pt_EB_s->Sumw2();
  h_pt_EB_b->Sumw2();
  h_nVtx_EB_s->Sumw2();
  h_nVtx_EB_b->Sumw2();
  h_eta_EB_s->Sumw2();
  h_eta_EB_b->Sumw2();
  h_rho_EB_s->Sumw2();
  h_rho_EB_b->Sumw2();

  t_EB_s->Draw("pt>>h_pt_EB_s",EB_s,"goff");
  t_EB_b->Draw("pt>>h_pt_EB_b",EB_b,"goff");
  t_EB_s->Draw("nvtx>>h_nVtx_EB_s",EB_s,"goff");
  t_EB_b->Draw("nvtx>>h_nVtx_EB_b",EB_b,"goff");
  t_EB_s->Draw("scEta>>h_eta_EB_s",EB_s,"goff");
  t_EB_b->Draw("scEta>>h_eta_EB_b",EB_b,"goff");
  t_EB_s->Draw("rho>>h_rho_EB_s",EB_s,"goff");
  t_EB_b->Draw("rho>>h_rho_EB_b",EB_b,"goff");

  //EB, mvares cut
  TH1F * h_pt_EB_Cut_s    = new TH1F ("h_pt_EB_Cut_s","h_pt_EB_Cut_s",25,0,250);
  TH1F * h_pt_EB_Cut_b    = new TH1F ("h_pt_EB_Cut_b","h_pt_EB_Cut_b",25,0,250);
  TH1F * h_nVtx_EB_Cut_s  = new TH1F ("h_nVtx_EB_Cut_s","h_nVtx_EB_Cut_s",80,0,80);
  TH1F * h_nVtx_EB_Cut_b  = new TH1F ("h_nVtx_EB_Cut_b","h_nVtx_EB_Cut_b",80,0,80);
  TH1F * h_eta_EB_Cut_s   = new TH1F ("h_eta_EB_Cut_s","h_eta_EB_Cut_s",50,-2.5,2.5);
  TH1F * h_eta_EB_Cut_b   = new TH1F ("h_eta_EB_Cut_b","h_eta_EB_Cut_b",50,-2.5,2.5);
  TH1F * h_rho_EB_Cut_s   = new TH1F ("h_rho_EB_Cut_s","h_rho_EB_Cut_s",50,0,50);
  TH1F * h_rho_EB_Cut_b   = new TH1F ("h_rho_EB_Cut_b","h_rho_EB_Cut_b",50,0,50);

  //WP 90% Low mass
  TCut EB_Cut_s = "(mvares>-0.05 && abs(scEta)<1.5 && pt > 18)";
  TCut EB_Cut_b = "(mvares>-0.05 && abs(scEta)<1.5 && pt > 18)";

  h_pt_EB_Cut_s->Sumw2();
  h_pt_EB_Cut_b->Sumw2();
  h_nVtx_EB_Cut_s->Sumw2();
  h_nVtx_EB_Cut_b->Sumw2();
  h_eta_EB_Cut_s->Sumw2();
  h_eta_EB_Cut_b->Sumw2();
  h_rho_EB_Cut_s->Sumw2();
  h_rho_EB_Cut_b->Sumw2();

  t_EB_s->Draw("pt>>h_pt_EB_Cut_s",EB_Cut_s,"goff");
  t_EB_b->Draw("pt>>h_pt_EB_Cut_b",EB_Cut_b,"goff");
  t_EB_s->Draw("nvtx>>h_nVtx_EB_Cut_s",EB_Cut_s,"goff");
  t_EB_b->Draw("nvtx>>h_nVtx_EB_Cut_b",EB_Cut_b,"goff");
  t_EB_s->Draw("scEta>>h_eta_EB_Cut_s",EB_Cut_s,"goff");
  t_EB_b->Draw("scEta>>h_eta_EB_Cut_b",EB_Cut_b,"goff");
  t_EB_s->Draw("rho>>h_rho_EB_Cut_s",EB_Cut_s,"goff");
  t_EB_b->Draw("rho>>h_rho_EB_Cut_b",EB_Cut_b,"goff");

  // Reading tree for EE
  TFile *file_EE = new TFile(fileName_EE.c_str());
  TTree *t_EE_s = (TTree*)file_EE->Get("promptPhotons");
  TTree *t_EE_b = (TTree*)file_EE->Get("fakePhotons");

  // EE, no mvares cut
  TH1F * h_pt_EE_s    = new TH1F ("h_pt_EE_s","h_pt_EE_s",25,0,250);
  TH1F * h_pt_EE_b    = new TH1F ("h_pt_EE_b","h_pt_EE_b",25,0,250);
  TH1F * h_nVtx_EE_s  = new TH1F ("h_nVtx_EE_s","h_nVtx_EE_s",80,0,80);
  TH1F * h_nVtx_EE_b  = new TH1F ("h_nVtx_EE_b","h_nVtx_EE_b",80,0,80);
  TH1F * h_eta_EE_s   = new TH1F ("h_eta_EE_s","h_eta_EE_s",50,-2.5,2.5);
  TH1F * h_eta_EE_b   = new TH1F ("h_eta_EE_b","h_eta_EE_b",50,-2.5,2.5);
  TH1F * h_rho_EE_s   = new TH1F ("h_rho_EE_s","h_rho_EE_s",50,0,50);
  TH1F * h_rho_EE_b   = new TH1F ("h_rho_EE_b","h_rho_EE_b",50,0,50);

  h_pt_EE_s->Sumw2();
  h_pt_EE_b->Sumw2();
  h_nVtx_EE_s->Sumw2();
  h_nVtx_EE_b->Sumw2();
  h_eta_EE_s->Sumw2();
  h_eta_EE_b->Sumw2();
  h_rho_EE_s->Sumw2();
  h_rho_EE_b->Sumw2();

  TCut EE_s = "(mvares>-1 && abs(scEta)>1.5 && pt > 18)";
  TCut EE_b = "(mvares>-1 && abs(scEta)>1.5 && pt > 18)";

  t_EE_s->Draw("pt>>h_pt_EE_s",EE_s,"goff");
  t_EE_b->Draw("pt>>h_pt_EE_b",EE_b,"goff");
  t_EE_s->Draw("nvtx>>h_nVtx_EE_s",EE_s,"goff");
  t_EE_b->Draw("nvtx>>h_nVtx_EE_b",EE_b,"goff");
  t_EE_s->Draw("scEta>>h_eta_EE_s",EE_s,"goff");
  t_EE_b->Draw("scEta>>h_eta_EE_b",EE_b,"goff");
  t_EE_s->Draw("rho>>h_rho_EE_s",EE_s,"goff");
  t_EE_b->Draw("rho>>h_rho_EE_b",EE_b,"goff");

  //EE, mvares cut
  TH1F * h_pt_EE_Cut_s    = new TH1F ("h_pt_EE_Cut_s","h_pt_EE_Cut_s",25,0,250);
  TH1F * h_pt_EE_Cut_b    = new TH1F ("h_pt_EE_Cut_b","h_pt_EE_Cut_b",25,0,250);
  TH1F * h_nVtx_EE_Cut_s  = new TH1F ("h_nVtx_EE_Cut_s","h_nVtx_EE_Cut_s",80,0,80);
  TH1F * h_nVtx_EE_Cut_b  = new TH1F ("h_nVtx_EE_Cut_b","h_nVtx_EE_Cut_b",80,0,80);
  TH1F * h_eta_EE_Cut_s   = new TH1F ("h_eta_EE_Cut_s","h_eta_EE_Cut_s",50,-2.5,2.5);
  TH1F * h_eta_EE_Cut_b   = new TH1F ("h_eta_EE_Cut_b","h_eta_EE_Cut_b",50,-2.5,2.5);
  TH1F * h_rho_EE_Cut_s   = new TH1F ("h_rho_EE_Cut_s","h_rho_EE_Cut_s",50,0,50);
  TH1F * h_rho_EE_Cut_b   = new TH1F ("h_rho_EE_Cut_b","h_rho_EE_Cut_b",50,0,50);

  //WP90%  for Low mass
  TCut EE_Cut_s = "(mvares>-0.22 && abs(scEta)>1.5 && pt > 18)";
  TCut EE_Cut_b = "(mvares>-0.22 && abs(scEta)>1.5 && pt > 18)";
  
  h_pt_EE_Cut_s->Sumw2();
  h_pt_EE_Cut_b->Sumw2();
  h_nVtx_EE_Cut_s->Sumw2();
  h_nVtx_EE_Cut_b->Sumw2();
  h_eta_EE_Cut_s->Sumw2();
  h_eta_EE_Cut_b->Sumw2();
  h_rho_EE_Cut_s->Sumw2();
  h_rho_EE_Cut_b->Sumw2();

  t_EE_s->Draw("pt>>h_pt_EE_Cut_s",EE_Cut_s,"goff");
  t_EE_b->Draw("pt>>h_pt_EE_Cut_b",EE_Cut_b,"goff");
  t_EE_s->Draw("nvtx>>h_nVtx_EE_Cut_s",EE_Cut_s,"goff");
  t_EE_b->Draw("nvtx>>h_nVtx_EE_Cut_b",EE_Cut_b,"goff");
  t_EE_s->Draw("scEta>>h_eta_EE_Cut_s",EE_Cut_s,"goff");
  t_EE_b->Draw("scEta>>h_eta_EE_Cut_b",EE_Cut_b,"goff");
  t_EE_s->Draw("rho>>h_rho_EE_Cut_s",EE_Cut_s,"goff");
  t_EE_b->Draw("rho>>h_rho_EE_Cut_b",EE_Cut_b,"goff");

  // effs EB
  TH1F * h_pt_EB_eff_s = new TH1F ("h_pt_EB_eff_s","h_pt_EB_eff_s",25,0,250);
  TH1F * h_pt_EB_eff_b = new TH1F ("h_pt_EB_eff_b","h_pt_EB_eff_b",25,0,250);
  TH1F * h_nVtx_EB_eff_s = new TH1F ("h_nVtx_EB_eff_s","h_nVtx_EB_eff_s",80,0,80);
  TH1F * h_nVtx_EB_eff_b = new TH1F ("h_nVtx_EB_eff_b","h_nVtx_EB_eff_b",80,0,80);
  TH1F * h_eta_EB_eff_s = new TH1F ("h_eta_EB_eff_s","h_eta_EB_eff_s",50,-2.5,2.5);
  TH1F * h_eta_EB_eff_b = new TH1F ("h_eta_EB_eff_b","h_eta_EB_eff_b",50,-2.5,2.5);
  TH1F * h_rho_EB_eff_s = new TH1F ("h_rho_EB_eff_s","h_rho_EB_eff_s",50,0,50);
  TH1F * h_rho_EB_eff_b = new TH1F ("h_rho_EB_eff_b","h_rho_EB_eff_b",50,0,50);

  h_pt_EB_eff_s->Sumw2();
  h_pt_EB_eff_b->Sumw2();
  h_nVtx_EB_eff_s->Sumw2();
  h_nVtx_EB_eff_b->Sumw2();
  h_eta_EB_eff_s->Sumw2();
  h_eta_EB_eff_b->Sumw2();
  h_rho_EB_eff_s->Sumw2();
  h_rho_EB_eff_b->Sumw2();

  h_pt_EB_eff_s = (TH1F*)h_pt_EB_Cut_s->Clone("h_pt_EB_eff_s");
  h_pt_EB_eff_s->Divide(h_pt_EB_s);
  h_pt_EB_eff_b = (TH1F*)h_pt_EB_Cut_b->Clone("h_pt_EB_eff_b");
  h_pt_EB_eff_b->Divide(h_pt_EB_b);
  h_nVtx_EB_eff_s = (TH1F*)h_nVtx_EB_Cut_s->Clone("h_nVtx_EB_eff_s");
  h_nVtx_EB_eff_s->Divide(h_nVtx_EB_s);
  h_nVtx_EB_eff_b = (TH1F*)h_nVtx_EB_Cut_b->Clone("h_nVtx_EB_eff_b");
  h_nVtx_EB_eff_b->Divide(h_nVtx_EB_b);
  h_eta_EB_eff_s = (TH1F*)h_eta_EB_Cut_s->Clone("h_eta_EB_eff_s");
  h_eta_EB_eff_s->Divide(h_eta_EB_s);
  h_eta_EB_eff_b = (TH1F*)h_eta_EB_Cut_b->Clone("h_eta_EB_eff_b");
  h_eta_EB_eff_b->Divide(h_eta_EB_b);
  h_rho_EB_eff_s = (TH1F*)h_rho_EB_Cut_s->Clone("h_rho_EB_eff_s");
  h_rho_EB_eff_s->Divide(h_rho_EB_s);
  h_rho_EB_eff_b = (TH1F*)h_rho_EB_Cut_b->Clone("h_rho_EB_eff_b");
  h_rho_EB_eff_b->Divide(h_rho_EB_b);

  // effs EE
  TH1F * h_pt_EE_eff_s = new TH1F ("h_pt_EE_eff_s","h_pt_EE_eff_s",25,0,250);
  TH1F * h_pt_EE_eff_b = new TH1F ("h_pt_EE_eff_b","h_pt_EE_eff_b",25,0,250);
  TH1F * h_nVtx_EE_eff_s = new TH1F ("h_nVtx_EE_eff_s","h_nVtx_EE_eff_s",80,0,80);
  TH1F * h_nVtx_EE_eff_b = new TH1F ("h_nVtx_EE_eff_b","h_nVtx_EE_eff_b",80,0,80);
  TH1F * h_eta_EE_eff_s = new TH1F ("h_eta_EE_eff_s","h_eta_EE_eff_s",50,-2.5,2.5);
  TH1F * h_eta_EE_eff_b = new TH1F ("h_eta_EE_eff_b","h_eta_EE_eff_b",50,-2.5,2.5);
  TH1F * h_rho_EE_eff_s = new TH1F ("h_rho_EE_eff_s","h_rho_EE_eff_s",50,0,50);
  TH1F * h_rho_EE_eff_b = new TH1F ("h_rho_EE_eff_b","h_rho_EE_eff_b",50,0,50);

  h_pt_EE_eff_s->Sumw2();
  h_pt_EE_eff_b->Sumw2();
  h_nVtx_EE_eff_s->Sumw2();
  h_nVtx_EE_eff_b->Sumw2();
  h_eta_EE_eff_s->Sumw2();
  h_eta_EE_eff_b->Sumw2();
  h_rho_EE_eff_s->Sumw2();
  h_rho_EE_eff_b->Sumw2();

  h_pt_EE_eff_s = (TH1F*)h_pt_EE_Cut_s->Clone("h_pt_EE_eff_s");
  h_pt_EE_eff_s->Divide(h_pt_EE_s);
  h_pt_EE_eff_b = (TH1F*)h_pt_EE_Cut_b->Clone("h_pt_EE_eff_b");
  h_pt_EE_eff_b->Divide(h_pt_EE_b);
  h_nVtx_EE_eff_s = (TH1F*)h_nVtx_EE_Cut_s->Clone("h_nVtx_EE_eff_s");
  h_nVtx_EE_eff_s->Divide(h_nVtx_EE_s);
  h_nVtx_EE_eff_b = (TH1F*)h_nVtx_EE_Cut_b->Clone("h_nVtx_EE_eff_b");
  h_nVtx_EE_eff_b->Divide(h_nVtx_EE_b);
  h_eta_EE_eff_s = (TH1F*)h_eta_EE_Cut_s->Clone("h_eta_EE_eff_s");
  h_eta_EE_eff_s->Divide(h_eta_EE_s);
  h_eta_EE_eff_b = (TH1F*)h_eta_EE_Cut_b->Clone("h_eta_EE_eff_b");
  h_eta_EE_eff_b->Divide(h_eta_EE_b);
  h_rho_EE_eff_s = (TH1F*)h_rho_EE_Cut_s->Clone("h_rho_EE_eff_s");
  h_rho_EE_eff_s->Divide(h_rho_EE_s);
  h_rho_EE_eff_b = (TH1F*)h_rho_EE_Cut_b->Clone("h_rho_EE_eff_b");
  h_rho_EE_eff_b->Divide(h_rho_EE_b);

  // plot eff histos:
  // eff vs pT, EB
  TLegend *leg = new TLegend(0.3,0.3,0.7,0.7);
  leg->SetBorderSize(0);
  leg->SetFillColor(0);
  leg->SetHeader(" EB :");

  TLegend *leg_EE = new TLegend(0.3,0.3,0.7,0.7);
  leg_EE->SetBorderSize(0);
  leg_EE->SetFillColor(0);
  leg_EE->SetHeader(" EE :");

  TCanvas * can_pt_eff_EB = new TCanvas("can_pt_eff_EB","can_pt_eff_EB",600,600);
  can_pt_eff_EB->SetGrid();

  h_pt_EB_eff_s->SetStats(0);
  h_pt_EB_eff_s->SetTitle("");
  h_pt_EB_eff_s->GetXaxis()->SetTitle("Pt (GeV)");
  h_pt_EB_eff_s->GetYaxis()->SetTitle("Efficiency");
  h_pt_EB_eff_s->GetYaxis()->SetRangeUser(0.,1.);
  h_pt_EB_eff_s->SetLineColor(kBlue);
  h_pt_EB_eff_s->SetMarkerStyle(20);
  h_pt_EB_eff_s->SetMarkerSize(0.8);
  h_pt_EB_eff_s->SetMarkerColor(kBlue);
  h_pt_EB_eff_s->Draw("P");

  h_pt_EB_eff_b->SetLineColor(kRed);
  h_pt_EB_eff_b->SetMarkerStyle(20);
  h_pt_EB_eff_b->SetMarkerSize(0.8);
  h_pt_EB_eff_b->SetMarkerColor(kRed);
  h_pt_EB_eff_b->Draw("Psame");

  leg->AddEntry(h_pt_EB_eff_s, "Signal: Fall17","P");
  leg->AddEntry(h_pt_EB_eff_b, "Bkg : Fall17","P");
  leg->Draw("same");

  TLatex *txt = new TLatex(0.2, 0.9, "");
  // txt->SetTextSize(0.05);
  txt->DrawLatexNDC(0.1, 0.91, "CMS #bf{#it{#scale[0.8]{Simulation Preliminary}}}");
  txt->DrawLatexNDC(0.76, 0.91, "#bf{13 TeV}");
  txt->Draw("same");

  can_pt_eff_EB->Update();
  can_pt_eff_EB->Modified();

  can_pt_eff_EB->SaveAs("phoId_eff_vs_pt_barrel_Hgg.pdf");
  can_pt_eff_EB->SaveAs("phoId_eff_vs_pt_barrel_Hgg.png");
  can_pt_eff_EB->SaveAs("phoId_eff_vs_pt_barrel_Hgg.root");

  // eff vs nVtx, EB
  TCanvas * can_nVtx_eff_EB = new TCanvas("can_nVtx_eff_EB","can_nVtx_eff_EB",600,600);
  can_nVtx_eff_EB->SetGrid();

  h_nVtx_EB_eff_s->SetStats(0);
  h_nVtx_EB_eff_s->SetTitle("");
  h_nVtx_EB_eff_s->GetXaxis()->SetTitle("Number of Vertices");
  h_nVtx_EB_eff_s->GetYaxis()->SetTitle("Efficiency");
  h_nVtx_EB_eff_s->GetYaxis()->SetRangeUser(0.,1.);
  h_nVtx_EB_eff_s->SetLineColor(kBlue);
  h_nVtx_EB_eff_s->SetMarkerStyle(20);
  h_nVtx_EB_eff_s->SetMarkerSize(0.8);
  h_nVtx_EB_eff_s->SetMarkerColor(kBlue);
  h_nVtx_EB_eff_s->Draw("P");

  h_nVtx_EB_eff_b->SetLineColor(kRed);
  h_nVtx_EB_eff_b->SetMarkerStyle(20);
  h_nVtx_EB_eff_b->SetMarkerSize(0.8);
  h_nVtx_EB_eff_b->SetMarkerColor(kRed);
  h_nVtx_EB_eff_b->Draw("Psame");

  leg->Draw("same");

  txt->DrawLatexNDC(0.1, 0.91, "CMS #bf{#it{#scale[0.8]{Simulation Preliminary}}}");
  txt->DrawLatexNDC(0.76, 0.91, "#bf{13 TeV}");
  txt->Draw("same");

  can_nVtx_eff_EB->Update();
  can_nVtx_eff_EB->Modified();

  can_nVtx_eff_EB->SaveAs("phoId_eff_vs_nVtx_barrel_Hgg.pdf");
  can_nVtx_eff_EB->SaveAs("phoId_eff_vs_nVtx_barrel_Hgg.png");
  can_nVtx_eff_EB->SaveAs("phoId_eff_vs_nVtx_barrel_Hgg.root");

  // eff vs eta, EB
  /* TCanvas * can_eta_eff_EB = new TCanvas("can_eta_eff_EB","can_eta_eff_EB",600,600);
  can_eta_eff_EB->SetGrid();

  h_eta_EB_eff_s->SetStats(0);
  h_eta_EB_eff_s->SetTitle("");
  h_eta_EB_eff_s->GetXaxis()->SetTitle("supercluster eta");
  h_eta_EB_eff_s->GetYaxis()->SetTitle("Efficiency");
  h_eta_EB_eff_s->GetYaxis()->SetRangeUser(0.,1.);
  h_eta_EB_eff_s->SetLineColor(kBlue);
  h_eta_EB_eff_s->SetMarkerStyle(20);
  h_eta_EB_eff_s->SetMarkerSize(0.8);
  h_eta_EB_eff_s->SetMarkerColor(kBlue);
  h_eta_EB_eff_s->Draw("P");

  h_eta_EB_old_eff_s->SetLineColor(kRed);
  h_eta_EB_old_eff_s->SetMarkerStyle(20);
  h_eta_EB_old_eff_s->SetMarkerSize(0.8);
  h_eta_EB_old_eff_s->SetMarkerColor(kRed);
  h_eta_EB_old_eff_s->Draw("Psame");

  h_eta_EB_eff_b->SetLineColor(kRed);
  h_eta_EB_eff_b->SetMarkerStyle(20);
  h_eta_EB_eff_b->SetMarkerSize(0.8);
  h_eta_EB_eff_b->SetMarkerColor(kRed);
  h_eta_EB_eff_b->Draw("Psame");

  h_eta_EB_old_eff_b->SetLineColor(kRed-9);
  h_eta_EB_old_eff_b->SetMarkerStyle(20);
  h_eta_EB_old_eff_b->SetMarkerSize(0.8);
  h_eta_EB_old_eff_b->SetMarkerColor(kRed-9);
  h_eta_EB_old_eff_b->Draw("Psame");

  leg->Draw("same");

  can_eta_eff_EB->Update();
  can_eta_eff_EB->Modified();

  can_eta_eff_EB->SaveAs("phoId_eff_vs_eta_barrel_Hgg.pdf");
  can_eta_eff_EB->SaveAs("phoId_eff_vs_eta_barrel_Hgg.png");
  can_eta_eff_EB->SaveAs("phoId_eff_vs_eta_barrel_Hgg.root"); */

  // eff vs pT, EE 
  TCanvas * can_pt_eff_EE = new TCanvas("can_pt_eff_EE","can_pt_eff_EE",600,600);
  can_pt_eff_EE->SetGrid();

  h_pt_EE_eff_s->SetStats(0);
  h_pt_EE_eff_s->SetTitle("");
  h_pt_EE_eff_s->GetXaxis()->SetTitle("Pt (GeV)");
  h_pt_EE_eff_s->GetYaxis()->SetTitle("Efficiency");
  h_pt_EE_eff_s->GetYaxis()->SetRangeUser(0.,1.);
  h_pt_EE_eff_s->SetLineColor(kBlue);
  h_pt_EE_eff_s->SetMarkerStyle(20);
  h_pt_EE_eff_s->SetMarkerSize(0.8);
  h_pt_EE_eff_s->SetMarkerColor(kBlue);
  h_pt_EE_eff_s->Draw("P");

  h_pt_EE_eff_b->SetLineColor(kRed);
  h_pt_EE_eff_b->SetMarkerStyle(20);
  h_pt_EE_eff_b->SetMarkerSize(0.8);
  h_pt_EE_eff_b->SetMarkerColor(kRed);
  h_pt_EE_eff_b->Draw("Psame");

  leg_EE->AddEntry(h_pt_EE_eff_s, "Signal: Fall17","P");
  leg_EE->AddEntry(h_pt_EE_eff_b, "Bkg : Fall17","P");

  leg_EE->Draw("same");

  txt->DrawLatexNDC(0.1, 0.91, "CMS #bf{#it{#scale[0.8]{Simulation Preliminary}}}");
  txt->DrawLatexNDC(0.76, 0.91, "#bf{13 TeV}");
  txt->Draw("same");

  can_pt_eff_EE->Update();
  can_pt_eff_EE->Modified();

  can_pt_eff_EE->SaveAs("phoId_eff_vs_pt_endcap_Hgg.pdf");
  can_pt_eff_EE->SaveAs("phoId_eff_vs_pt_endcap_Hgg.png");
  can_pt_eff_EE->SaveAs("phoId_eff_vs_pt_endcap_Hgg.root");

  // eff vs nVtx, EE
  TCanvas * can_nVtx_eff_EE = new TCanvas("can_nVtx_eff_EE","can_nVtx_eff_EE",600,600);
  can_nVtx_eff_EE->SetGrid();

  h_nVtx_EE_eff_s->SetStats(0);
  h_nVtx_EE_eff_s->SetTitle("");
  h_nVtx_EE_eff_s->GetXaxis()->SetTitle("Number of Vertices");
  h_nVtx_EE_eff_s->GetYaxis()->SetTitle("Efficiency");
  h_nVtx_EE_eff_s->GetYaxis()->SetRangeUser(0.,1.);
  h_nVtx_EE_eff_s->SetLineColor(kBlue);
  h_nVtx_EE_eff_s->SetMarkerStyle(20);
  h_nVtx_EE_eff_s->SetMarkerSize(0.8);
  h_nVtx_EE_eff_s->SetMarkerColor(kBlue);
  h_nVtx_EE_eff_s->Draw("P");

  h_nVtx_EE_eff_b->SetLineColor(kRed);
  h_nVtx_EE_eff_b->SetMarkerStyle(20);
  h_nVtx_EE_eff_b->SetMarkerSize(0.8);
  h_nVtx_EE_eff_b->SetMarkerColor(kRed);
  h_nVtx_EE_eff_b->Draw("Psame");

  leg_EE->Draw("same");

  txt->DrawLatexNDC(0.1, 0.91, "CMS #bf{#it{#scale[0.8]{Simulation Preliminary}}}");
  txt->DrawLatexNDC(0.76, 0.91, "#bf{13 TeV}");
  txt->Draw("same");

  can_nVtx_eff_EE->Update();
  can_nVtx_eff_EE->Modified();

  can_nVtx_eff_EE->SaveAs("phoId_eff_vs_nVtx_endcap_Hgg.pdf");
  can_nVtx_eff_EE->SaveAs("phoId_eff_vs_nVtx_endcap_Hgg.png");
  can_nVtx_eff_EE->SaveAs("phoId_eff_vs_nVtx_endcap_Hgg.root");

  // eff vs rho, EB
  TCanvas * can_rho_eff_EB = new TCanvas("can_rho_eff_EB","can_rho_eff_EB",600,600);
  can_rho_eff_EB->SetGrid();

  h_rho_EB_eff_s->SetStats(0);
  h_rho_EB_eff_s->SetTitle("");
  h_rho_EB_eff_s->GetXaxis()->SetTitle("#rho");
  h_rho_EB_eff_s->GetYaxis()->SetTitle("Efficiency");
  h_rho_EB_eff_s->GetYaxis()->SetRangeUser(0.,1.);
  h_rho_EB_eff_s->SetLineColor(kBlue);
  h_rho_EB_eff_s->SetMarkerStyle(20);
  h_rho_EB_eff_s->SetMarkerSize(0.8);
  h_rho_EB_eff_s->SetMarkerColor(kBlue);
  h_rho_EB_eff_s->Draw("P");

  h_rho_EB_eff_b->SetLineColor(kRed);
  h_rho_EB_eff_b->SetMarkerStyle(20);
  h_rho_EB_eff_b->SetMarkerSize(0.8);
  h_rho_EB_eff_b->SetMarkerColor(kRed);
  h_rho_EB_eff_b->Draw("Psame");

  leg->Draw("same");

  txt->DrawLatexNDC(0.1, 0.91, "CMS #bf{#it{#scale[0.8]{Simulation Preliminary}}}");
  txt->DrawLatexNDC(0.76, 0.91, "#bf{13 TeV}");
  txt->Draw("same");

  can_rho_eff_EB->Update();
  can_rho_eff_EB->Modified();

  can_rho_eff_EB->SaveAs("phoId_eff_vs_rho_barrel_Hgg.pdf");
  can_rho_eff_EB->SaveAs("phoId_eff_vs_rho_barrel_Hgg.png");
  can_rho_eff_EB->SaveAs("phoId_eff_vs_rho_barrel_Hgg.root");

  // eff vs rho, EE 
  TCanvas * can_rho_eff_EE = new TCanvas("can_rho_eff_EE","can_rho_eff_EE",600,600);
  can_rho_eff_EE->SetGrid();

  h_rho_EE_eff_s->SetStats(0);
  h_rho_EE_eff_s->SetTitle("");
  h_rho_EE_eff_s->GetXaxis()->SetTitle("#rho");
  h_rho_EE_eff_s->GetYaxis()->SetTitle("Efficiency");
  h_rho_EE_eff_s->GetYaxis()->SetRangeUser(0.,1.);
  h_rho_EE_eff_s->SetLineColor(kBlue);
  h_rho_EE_eff_s->SetMarkerStyle(20);
  h_rho_EE_eff_s->SetMarkerSize(0.8);
  h_rho_EE_eff_s->SetMarkerColor(kBlue);
  h_rho_EE_eff_s->Draw("P");

  h_rho_EE_eff_b->SetLineColor(kRed);
  h_rho_EE_eff_b->SetMarkerStyle(20);
  h_rho_EE_eff_b->SetMarkerSize(0.8);
  h_rho_EE_eff_b->SetMarkerColor(kRed);
  h_rho_EE_eff_b->Draw("Psame");

  leg_EE->Draw("same");

  txt->DrawLatexNDC(0.1, 0.91, "CMS #bf{#it{#scale[0.8]{Simulation Preliminary}}}");
  txt->DrawLatexNDC(0.76, 0.91, "#bf{13 TeV}");
  txt->Draw("same");

  can_rho_eff_EE->Update();
  can_rho_eff_EE->Modified();

  can_rho_eff_EE->SaveAs("phoId_eff_vs_rho_endcap_Hgg.pdf");
  can_rho_eff_EE->SaveAs("phoId_eff_vs_rho_endcap_Hgg.png");
  can_rho_eff_EE->SaveAs("phoId_eff_vs_rho_endcap_Hgg.root");

  // eff vs eta
  TCanvas * can_eta_eff = new TCanvas("can_eta_eff","can_eta_eff",600,600);
  can_eta_eff->SetGrid();

  h_eta_EB_eff_s->SetStats(0);
  h_eta_EB_eff_s->SetTitle("");
  h_eta_EB_eff_s->GetXaxis()->SetTitle("#eta_{SC}");
  h_eta_EB_eff_s->GetYaxis()->SetTitle("Efficiency");
  h_eta_EB_eff_s->GetYaxis()->SetRangeUser(0.,1.);
  h_eta_EB_eff_s->SetLineColor(kBlue);
  h_eta_EB_eff_s->SetMarkerStyle(20);
  h_eta_EB_eff_s->SetMarkerSize(0.8);
  h_eta_EB_eff_s->SetMarkerColor(kBlue);
  h_eta_EB_eff_s->Draw("P");

  h_eta_EB_eff_b->SetLineColor(kRed);
  h_eta_EB_eff_b->SetMarkerStyle(20);
  h_eta_EB_eff_b->SetMarkerSize(0.8);
  h_eta_EB_eff_b->SetMarkerColor(kRed);
  h_eta_EB_eff_b->Draw("Psame");

  h_eta_EE_eff_s->SetLineColor(kBlue);
  h_eta_EE_eff_s->SetMarkerStyle(20);
  h_eta_EE_eff_s->SetMarkerSize(0.8);
  h_eta_EE_eff_s->SetMarkerColor(kBlue);
  h_eta_EE_eff_s->Draw("Psame");

  h_eta_EE_eff_b->SetLineColor(kRed);
  h_eta_EE_eff_b->SetMarkerStyle(20);
  h_eta_EE_eff_b->SetMarkerSize(0.8);
  h_eta_EE_eff_b->SetMarkerColor(kRed);
  h_eta_EE_eff_b->Draw("Psame");

  leg->SetHeader("");
  leg->Draw("same");

  txt->DrawLatexNDC(0.1, 0.91, "CMS #bf{#it{#scale[0.8]{Simulation Preliminary}}}");
  txt->DrawLatexNDC(0.76, 0.91, "#bf{13 TeV}");
  txt->Draw("same");

  can_eta_eff->Update();
  can_eta_eff->Modified();

  can_eta_eff->SaveAs("phoId_eff_vs_eta_Hgg.pdf");
  can_eta_eff->SaveAs("phoId_eff_vs_eta_Hgg.png");
  can_eta_eff->SaveAs("phoId_eff_vs_eta_Hgg.root");

}
