function tspec_helcorr, header, debug=debug
;
; Derive heliocentric correction and apply it to the TripleSpec data
; requirements: helcorr(), date_conv()
;
  if ~keyword_set(debug) then debug=0

; Convert date to julian day using header keyword
  date = fxpar(header, "DATE-OBS")
  jd   = double(date_conv( date, 'Julian'))
  
  if jd le 0 then stop, "  <ERROR> Problems on JD calculation. Check DATE-OBS info on the header."

  ra = tenv(fxpar(header, "RA" )) ; in hours
  de = tenv(fxpar(header, "DEC")); in deg

; SOAR telescope coordinates
  ;obs_lat = double(ten(-30, 14, 16.41)) ; S < 0
  ;obs_lon = double(ten(70, 44, 01.11))  ; W > 0 
  ;obs_alt = double(2748)                ; in m
  obs_lat = double(-30.238) ; S < 0
  obs_lon = double(+70.734)  ; W > 0 
  obs_alt = double(2738)                ; in m
  
; calculates the heliocentric correction
  helcorr, obs_lon, obs_lat, obs_alt, ra, de, jd, vcorr, hjd, debug=keyword_set(debug)
  
; apply the correction on the wave_cor = wavelength * ( 1. + vcorr / 2.99792E5 )
  return, vcorr

end


pro triplespec_spectroimages, sci_on=sci_on, sci_off=sci_off, flat_sol=flat_sol, arc_sol=arc_sol, $
                              off_ratio=off_ratio, flux_sol=flux_sol, outfolder=outfolder, helio_corr=helio_corr, $
                              reverse_slit=reverse_slit, plot_ci=plot_ci

;+
; NAME:
;       TRIPLESPEC_SPECTROIMAGES
; PURPOSE:
;       Converts TripleSpec/SOAR cross-dispersed data into linearized spectroimages
;       in both spectral and spatial directions
; EXPLANATION:
;       Use TripleSpec raw images and Spextool by-products to deliver spectroimages
;       of orders 3 to 7 in a linear scale in both axis
; CALLING SEQUENCE:
;       triplespec_spectroimages, sci_on = , flat_sol = , arc_sol = , outfolder = ,$
;                        [sci_off = , off_frac = , flux_sol = , plot_CI = , /helio_corr ]
;
; INPUTS:
;       sci_on    = raw ".fits" file corresponding to the on-source frame
;       flat_sol  = Spextool product containing the normalized flat field solution
;                   (in general, located in the CAL folder).
;       arc_sol   = Spextool product containing the wavelength solution from the comparison lamp spectrum
;                   (in general, located in the CAL folder).
;       outfolder = full path for saving the output files
;
; OPTIONAL INPUTS:
;       sci_off   = raw ".fits" file corresponding to the off-source frame
;                   (if provided, sky-subtraction is performed automatically)
;       off_ratio = multiply the OFF source to improve sky subtraction (default=1.0)
;       flux_sol  = Spextool product containing the flux calibration 1d spectrum (from 'xtellcor' procedure)
;       plot_CI   = Confidence Interval to plot the data (from 0 to 1, default=0.95)
;       helio_corr= set to 1 if you want to perform heliocentric correction (default: 0)
;       reverse_slit = set to True to reverse slit orientation
;
; OUTPUTS:
;       Function result = coefficient vector. 
;       If = 0.0 (scalar), no fit was possible.
;       If vector has more than 2 elements (the last=0) then the fit is dubious.
;
; SUBROUTINE CALLS:
;       triplespec_helcorr()
;       fits_read()
;       sxpar()
;       mkhdr()
;       fits_write()
;       cgimage()
;
; REVISION HISTORY:
;       2022-08-01: Written by F. Navarete (RSS/NOIRLab)
;       2022-09-10: Transformed into a callable function to be sexecuted on IDL terminal.
;       2022-09-13: Add saturation information on the header
;       2022-10-06: Removed reversed y-axis, now the slit position matches the output from spextool.
;-------------------------------------------------------

; check if the minimum parameters were set
if keyword_set(arc_sol)+keyword_set(flat_sol)+keyword_set(sci_on)+keyword_set(outfolder) LT 4 then begin
  print,"  <ERROR> Minimum function call:"
  stop, "          tspec_extract2d, sci_on=sci_on, flat_sol=flat_sol, arc_sol=arc_sol, outfolder=outfolder"
endif

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Check if input files exist
  if ~file_test(arc_sol)     then stop, "  <ERROR> 'arc_sol' does not exist."
  if ~file_test(flat_sol)    then stop, "  <ERROR> 'flat_sol' does not exist."
  if ~file_test(sci_on)      then stop, "  <ERROR> 'sci_on' does not exist."
  if ~keyword_set(outfolder) then stop, "  <ERROR> 'outfolder' must be provided."

; Check if optional input files exist
  if keyword_set(sci_off)    then $
    if ~file_test(sci_off)   then stop, "  <ERROR> 'sci_off' does not exist."
  if keyword_set(flux_sol)   then $
    if ~file_test(flux_sol)  then stop, "  <ERROR> 'flux_sol' does not exist."

; if outfolder does not exist, create it
  if ~file_test(outfolder)   then file_mkdir, outfolder

; default values for optional parameters
  if ~keyword_set(off_ratio) then off_ratio = 1.0
  if ~keyword_set(plot_ci)   then plot_ci   = 0.95
; define saturation limit (in DN) for TripleSpec
  saturation_limit = 30000.00
  
; open fits files
  fits_read, arc_sol,  junk, hdr_arc, exten_no=0
  fits_read, arc_sol,  img_wave, exten_no=1
  fits_read, arc_sol,  img_spat, exten_no=2
  ; check if the 'arc_sol' contains the required keywords
  if sxpar(hdr_arc,'DISPO03') LE 0 then stop,"  <ERROR> 'arc_sol' does not have the required keywords. check input file."
  
  fits_read, flat_sol,  junk, hdr_flat, exten_no=0
  fits_read, flat_sol, img_flat, exten_no=1
  if sxpar(hdr_flat,'PLTSCALE') LE 0 then stop,"  <ERROR> 'flat_sol' does not have the required keywords. check input file."
  
  fits_read, sci_on,   img_a,    hdr_a
  if keyword_set(sci_off)  then fits_read, sci_off, img_b, hdr_b
  if keyword_set(flux_sol) then fits_read, flux_sol, flux_img, hdr_flux


; the dispersion for each order is given in the header of the arc solution
  dispersion = [ sxpar(hdr_arc,'DISPO03'), sxpar(hdr_arc,'DISPO04'), sxpar(hdr_arc,'DISPO05'), sxpar(hdr_arc,'DISPO06'), sxpar(hdr_arc,'DISPO07')]
  
  
; flat correct the input AB images
  img_a /= img_flat
  
  if keyword_set(sci_off) then begin
    img_b /= img_flat
  ; make A-B
    img_sci = img_a - ( img_b * off_ratio )
  endif else img_sci = img_a

; Save current device name
  OS_device = !D.NAME
  set_plot,'PS'

; loop into orders 3 to 7
for order=3, 7 do begin  
  
  if order EQ 3 then x_order=[874,979] ; K band
  if order EQ 4 then x_order=[704,814] ; H band
  if order EQ 5 then x_order=[529,684] ; J band
  if order EQ 6 then x_order=[319,554] ; Y band
  if order EQ 7 then x_order=[245,420] ; Y band (blue)

  cut_wave = img_wave[x_order[0]:x_order[1],*]
  cut_spat = img_spat[x_order[0]:x_order[1],*]
  cut_sci  = img_sci[x_order[0]:x_order[1],*]
  
; for order 7, crop data on y-axis
  if order eq 7 then begin
    cut_wave = cut_wave[*,1029:2047]
    cut_spat = cut_spat[*,1029:2047]
    cut_sci  =  cut_sci[*,1029:2047]
  endif

; exclude traces from adjascent orders (to be sure we're properly fitting the actual order)
  if order EQ 5 then cut_wave[0:24,1389:2047] = !Values.F_NaN ; exclude tracing from other orders
  if order EQ 6 then cut_wave[210:235,0:379]  = !Values.F_NaN ; exclude tracing from other orders
  if order EQ 6 then cut_wave[0:94,979:2047]  = !Values.F_NaN ; exclude tracing from other orders
  if order EQ 7 then cut_wave[154:175,0:200]  = !Values.F_NaN ; exclude tracing from other orders
  
  if order EQ 5 then cut_spat[0:24,1389:2047] = !Values.F_NaN ; exclude tracing from other orders
  if order EQ 6 then cut_spat[210:235,0:379]  = !Values.F_NaN ; exclude tracing from other orders
  if order EQ 6 then cut_spat[0:94,979:2047]  = !Values.F_NaN ; exclude tracing from other orders
  if order EQ 7 then cut_spat[154:175,0:200]  = !Values.F_NaN ; exclude tracing from other orders
  
  if order EQ 5 then cut_sci[0:24,1389:2047]  = !Values.F_NaN ; exclude tracing from other orders
  if order EQ 6 then cut_sci[210:235,0:379]   = !Values.F_NaN ; exclude tracing from other orders
  if order EQ 6 then cut_sci[0:94,979:2047]   = !Values.F_NaN ; exclude tracing from other orders
  if order EQ 7 then cut_sci[154:175,0:200]   = !Values.F_NaN ; exclude tracing from other orders
  
; get size of the images of the current order
  img_size = size(cut_sci, /dim)
  
; loop into dispersion axis - this will search for the minimum plate scale along the slit: ~0.35"/pix.
; I will use the value on header - sxpar(hdr_flat,'PLTSCALE')
  for y=0, img_size[1]-1 do begin
    idx_fin = where( finite(cut_spat[*,y]), n_fin )
    if n_fin gt 0 then begin
      sl = cut_spat[idx_fin,y]
      dl = sl[0:n_elements(sl)-2]-sl[1:n_elements(sl)-1]
      mindisp_y = min(dl)
  
      spat_range_y = [ min(sl), max(sl) ]
      if ~keyword_set(spat_range) then spat_range=spat_range_y else begin
        if min(spat_range_y) LT min(spat_range) then spat_range[0] = min(spat_range_y)
        if max(spat_range_y) GT max(spat_range) then spat_range[1] = max(spat_range_y)
      endelse
      
      if ~keyword_set(mindisp) then mindisp=mindisp_y else $
        if mindisp_y LT mindisp then mindisp = mindisp_y
    endif
  
  endfor

; now create uniform array on the spatial axis
  spat_disp = sxpar(hdr_flat,'PLTSCALE')
  spat_range = [ min(spat_range), max(spat_range) ]
  ny = round( ( spat_range[1] - spat_range[0] ) / spat_disp ) + 1
  
  ; I will interpolate each line using this linear grid
  sy = findgen(ny) * spat_disp + spat_range[0]

; now interpolate the data on the y-direction using the linear 'sx' scale
  interp_sci = fltarr(ny,img_size[1])
  interp_wave = fltarr(ny,img_size[1])
  for y=0, img_size[1]-1 do begin
    idx_fin = where( finite(cut_spat[*,y]), n_fin )
    if n_fin gt 0 then begin
      sl = cut_spat[idx_fin,y]
      sli = interpol(cut_sci[idx_fin,y],sl,sy)
      interp_sci[*,y] = sli

      swi = interpol(cut_wave[idx_fin,y],sl,sy)
      interp_wave[*,y] = swi
    endif
  endfor
  
; now create uniform array on the dispersion axis
  idx_fin = where( interp_wave GT 0, n_fin )
  spec_disp = dispersion[order-3]
  nx = round( ( max(interp_wave[idx_fin],/nan) - min(interp_wave[idx_fin],/nan) ) / spec_disp )
  if order ne 7 then nx = 2040 ; for orders 3 to 6, the dispersion axis size should be fixed on 2040 pixels
  
  sw = findgen(nx) * spec_disp + min(interp_wave[idx_fin],/nan)
  
  interp_sw_sci  = fltarr(ny,nx)
  
; flux calibrate the data
  if keyword_set(flux_sol) then begin
    wflux = flux_img[*,0,order-3]
    fflux = flux_img[*,1,order-3]
    idxf  = where( finite(fflux) and finite(wflux) )
    fintp = interpol( fflux[idxf], wflux[idxf], sw )
  ; estimate the saturation limit in flux units
    sat_flux = min(saturation_limit * fflux[idxf])
  endif else sat_flux = saturation_limit

; apply heliocentric velocity correction
; wave_cor = wavelength * ( 1. + vhelio / 2.99792E5 )
  if keyword_set(helio_corr) then begin
    vhelio = tspec_helcorr(hdr_a)
    sw_cor = sw * ( 1. + vhelio / 2.99792E5 )
  endif

  for y=0, ny-1 do begin
    ;idx_fin = where( interp_sci[y,*] NE 0, n_fin ) ; moved from EQ 0 to NE 0 so it can properly deal with A-B cases
    ;if n_fin gt 0 then begin
    ;swx = interp_wave[y,idx_fin]
    swx = interp_wave[y,*]
    ;swi = interpol(interp_sci[y,idx_fin],swx,sw)
    swi = interpol(interp_sci[y,*],swx,sw)
    
    if keyword_set(helio_corr) then  begin
      swic = interpol(swi,sw_cor,sw)
      swi = swic
    endif
    if keyword_set(flux_sol) then swi *= fintp ; (TBD)
    interp_sw_sci[y,*] = swi
    ;endif
  endfor
  
; transpose the array so x and y are spectral and spatial dimensions, respectively
  img_out = transpose(interp_sw_sci)
  
  if keyword_set(reverse_slit) then $
    img_out = REVERSE(img_out, 2) 
  
; truncate pixels above saturation limit threshold
  idx_sat = where(img_out GE sat_flux, n_sat)
  if n_sat GT 0 then img_out[idx_sat] = sat_flux

; divide by exposure time 
  exptime = sxpar(hdr_a,'EXPTIME')
  ; flux is given as Wm-2um-1/DNs-1
  ; flux_dn / exptime * cal leads to: W m-2 um-1
  img_out /= exptime
  sat_flux /= exptime

  sy_out = sy
  
; make output header
  mkhdr, header_out, img_out
  
; copy keywords from raw data
  kw=['OBJECT', 'PROPID', 'NIGHTID', 'OBSID', 'FILENAME', 'OBSMODE', 'OBSERVER', 'INSTRUM', 'OBSTYPE', 'DATE-OBS', $
      'FSAMPLE', 'COADDS', 'RON', 'GAIN', 'OBSERVAT', 'TELESCOP', 'MJD', 'UT', 'DATE', 'RA', 'DEC', 'HA', $
      'TELAZ', 'TELEL', 'SIDEREAL', 'AIRMASS', 'IPA', 'PARALL']
  for k=0, n_elements(kw)-1 do $
    sxaddpar, header_out, kw[k], sxpar(hdr_a, kw[k], comment=comment), comment
  
; add axis information  
  sxaddpar, header_out, 'CRVAL1',sw[0]
  sxaddpar, header_out, 'CRPIX1',1
  sxaddpar, header_out, 'CDELT1',sw[1]-sw[0]
  sxaddpar, header_out, 'CRVAL2',sy_out[0]
  sxaddpar, header_out, 'CRPIX2',1
  sxaddpar, header_out, 'CDELT2',sy_out[1]-sy_out[0]

; exptime should be equal to 1.0 
  sxaddpar, header_out, 'EXPTIME', 1.0 , 'Flux is per second'
  sxaddpar, header_out, 'EXPTIME0', sxpar(hdr_a,'EXPTIME', comment=comment), 'Original '+comment

; create required keywords for Skycorr/ESO
  date_str=  sxpar(hdr_a,'date-obs')
    year = Fix(StrMid(date_str,0,4))
    mon =  Fix(StrMid(date_str,5,2))
    day =  Fix(StrMid(date_str,8,2))
    hour = Fix(StrMid(date_str,11,2))
    min =  Fix(StrMid(date_str,14,2))
    sec =  float(StrMid(date_str,17,6))
      jultime = JulDay(mon,day,year,hour, min,sec)
      mjd_obs = jultime - 2400000.5
      ut_sec = hour * 3600. + min * 60 + sec  
  sxaddpar, header_out, 'MJD-OBS', mjd_obs, 'MJD at start of the observation'
  sxaddpar, header_out, 'TM-START', ut_sec, 'UT at start of the observation (in seconds)'
  
; specific keywords for this script
  root_on = strmid(sci_on, strpos(sci_on, '/',/reverse_search)+1,strpos(sci_on, '.fits',/reverse_search)-strpos(sci_on, '/',/reverse_search)-1)
  if keyword_set(sci_off) then begin
    root_off= strmid(sci_off,strpos(sci_off,'/',/reverse_search)+1,strpos(sci_off,'.fits',/reverse_search)-strpos(sci_off,'/',/reverse_search)-1)
    sxaddpar, header_out, 'E2D_MODE','ON-OFF', 'ON-OFF or ON'
    sxaddpar, header_out, 'E2D_ON',root_on, 'ON image'
    sxaddpar, header_out, 'E2D_OFF',root_off, 'OFF image'
  endif else begin
    sxaddpar, header_out, 'E2D_MODE','ON', 'ON-OFF or ON'
    sxaddpar, header_out, 'E2D_ON',root_on, 'ON image'
  endelse  
; heliocentric correction  
  if keyword_set(helio_corr) then $
    sxaddpar, header_out, 'E2D_VHEL', vhelio, 'Heliocentric velocity correction (in km/s)'
; flux calibration
  if keyword_set(flux_sol) then flag_fluxcor = 'True' $
                           else flag_fluxcor = 'False'
  sxaddpar, header_out, 'E2D_FLUX',flag_fluxcor, 'True if flux calibration was applied.'
  if keyword_set(flux_sol) then sxaddpar, header_out, 'YUNITS', sxpar(hdr_flux,'YUNITS', comment=comment), 'Flux units (per DN/sec)' $
                           else sxaddpar, header_out, 'YUNITS', 'DN/sec', 'Flux units (per DN/sec)'
  ;if keyword_set(flux_sol) then sxaddpar, header_out, 'E2D_SAT', sat_flux, 'Saturation limit (in YUNITS units)' $
  ;                         else sxaddpar, header_out, 'E2D_SAT', saturation_limit/exptime, 'Saturation limit (in DN/sec)'
  sxaddpar, header_out, 'E2D_SAT', sat_flux, 'Saturation limit (in YUNITS units)'
  sxaddpar, header_out, 'E2D_NSAT', n_sat,   'Number of saturated pixels'

; create output names based on the ON-Source file
  output = strmid(sci_on,strpos(sci_on,'/',/reverse_search)+1,strpos(sci_on,'.fits',/reverse_search)-strpos(sci_on,'/',/reverse_search)-1)
  output_fits = output + '_2d_' + strtrim(order,2) + '.fits'
  output_ps   = output + '_2d_' + strtrim(order,2) + '.ps'

; write the .fits file  
  fits_write, outfolder + output_fits, img_out, header_out

; now create the output ps file  
  device,file=outfolder + output_ps, xsize=20, ysize=8, font_size=7
  ; set position of the image and the colorbar
    pos_img = [0.075,0.15,0.89,0.9]
    pos_bar = [0.90,0.15,0.915,0.9]

  ; use the 95% percentile range to display the data
    conflimit=plot_ci
      lowindex = long(((1.0-conflimit)/2)*n_elements(img_out))
      highindex = n_elements(img_out)-lowindex-1
      sortvalues = img_out[sort(img_out)]
    range_conf = [sortvalues[lowindex],sortvalues[highindex]]

  ; set the title    
    case order of
      3: title_order = '(K-band, n=3)'
      4: title_order = '(H-band, n=4)'
      5: title_order = '(J-band, n=5)'
      6: title_order = '(Y-band, n=6)'
      7: title_order = '(Y-band, n=7)'
    endcase
    title_order = sxpar(hdr_a,'OBJECT') + ' ' + title_order

  ; plot the image
    cgimage, smooth(img_out,0,/edge_truncate), xrange=[min(sw),max(sw)], yrange=[min(sy_out),max(sy_out)], /axis, pos=pos_img, $
             minval=range_conf[0],maxval=range_conf[1], ctindex=13, $
             ytitle='Slit direction (arcsec)', xtitle=('Wavelength (um)'), title=title_order, axkeywords={charthick:2}

  ; place the colorbar
    cgcolorbar, /vertical, /right, ctindex=13, minrange=range_conf[0],maxrange=range_conf[1], pos=pos_bar, $
                title=sxpar(header_out,'YUNITS'), _REF_EXTRA={charthick:2}

  device, /close
  
endfor ; order=3,7

; now recover the standard OS device
  set_plot, OS_device
  
print, '  <END> Check output files on ' + outfolder
end