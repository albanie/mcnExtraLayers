function loadedImdb = fetchExternalImdb(imdbPath)
%FETCHEXTERNALIMDB - a helper function to avoid painful load times
%  LOADEDIMDB = FETCHEXTERNALIMDB(IMDBPATH) loads the imdb at the
%  given IMDBPATH from a global memory cache if it is available. If
%  not, the imdb is loaded from disk. In both cases, the loaded imdb
%  structure is returned as LOADEDIMDB.
% with external imdbs

	global gImdb ; % cache
  global gImdbPath ; % validate the the correct imdb is being loaded

  if strcmp(gImdbPath, imdbPath) && ~isempty(gImdb)
		fprintf('found imdb in cache, re-using..\n') ;
	else
		fprintf('loading imdb ...') ; tic ;
    gImdb = load(imdbPath) ;
		fprintf('done in %g s\n', toc) ;
	end
	loadedImdb = gImdb ;
