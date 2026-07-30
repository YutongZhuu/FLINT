if {$argc != 2} {
  puts stderr "usage: post_route_reports.tcl <routed.dcp> <report-directory>"
  exit 2
}

set routed_checkpoint [file normalize [lindex $argv 0]]
set report_dir [file normalize [lindex $argv 1]]

if {![file exists $routed_checkpoint]} {
  puts stderr "error: routed checkpoint does not exist: $routed_checkpoint"
  exit 2
}

file mkdir $report_dir
open_checkpoint $routed_checkpoint

report_timing_summary \
  -file [file join $report_dir timing_summary.rpt]
report_utilization \
  -file [file join $report_dir utilization.rpt]
report_utilization \
  -hierarchical \
  -hierarchical_depth 6 \
  -file [file join $report_dir utilization_hierarchical.rpt]
report_power \
  -file [file join $report_dir power.rpt]

close_design
