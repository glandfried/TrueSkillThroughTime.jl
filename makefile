all:
	echo "README\n"

push: 
	git add -f .
	git commit -m "Update"
	git push origin gh-pages
	
TrueSkillThroughTime.jl/.git:
	git submodule update --init TrueSkillThroughTime.jl/

crear: TrueSkillThroughTime.jl/.git 
	make -C TrueSkillThroughTime.jl/docs/
	
publicar: crear
	rsync -avrc --delete --exclude=.git --exclude=.gitmodules --exclude=makefile --exclude=TrueSkillThroughTime.jl/ TrueSkillThroughTime.jl/docs/build/ ./ 
	
	
