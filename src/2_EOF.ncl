begin
  name_file = "daily_anoma_window_smoothed15daysw_SMrz_GLEAM_AprMay_US.nc"
  filee = addfile("/scratch/negishi/castanev/GLEAM/" + "/" + name_file, "r")

  n = filee->SMrz(:,:,:)   ; (time, lat, lon); time is dim 0

  ; Keep dates as integers; strip string time coord (netCDF3 cannot write it)
  time_ymd = stringtointeger(tostring(filee->time))
  if (iscoord(n, "time")) then
    delete(n&time)
  end if

  neval  = 5               ; number of EOFs / PCs to return
  optEOF = False
  optETS = False

  eof_n  = eofunc_n_Wrap(n, neval, optEOF, 0)       ; spatial patterns (neval, lat, lon)
  eof_ts = eofunc_ts_n_Wrap(n, eof_n, optETS, 0)    ; PCs (neval, time)

  ; Ensure no string coordinate rides along on the PCs
  if (iscoord(eof_ts, "time")) then
    delete(eof_ts&time)
  end if
  eof_ts!1 = "time"
  time_ymd!0 = "time"

  out_path = "/scratch/negishi/castanev/GLEAM/EOF_timeseries_daily_anoma_window_smoothed15daysw_SMrz_GLEAM_AprMay_US.nc"
  system("rm -f " + out_path)   ; "c" fails if the file already exists
  oh = addfile(out_path, "c")
  oh->n_eof    = eof_n
  oh->n_pc     = eof_ts
  oh->time_ymd = time_ymd

  print("% var="+ eof_n@pcvar)

end
exit

