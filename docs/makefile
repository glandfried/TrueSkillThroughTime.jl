crear:
	julia make.jl

publicar: crear web/.git
	rsync -avrc --delete --exclude=.git --exclude=makefile build/ web/
	make -C web push
	
web/.git:
	git submodule update --init web
	cd web
	git checkout gh-pages
