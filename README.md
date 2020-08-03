
## Getting Started

### Running the app locally

First create a virtual environment with conda or venv inside a temp folder, then activate it.

- 第一次虚拟环境建立
```
#pip install virtualenv
cd "D:\\ck\\yzh\\2 replenishment\\8.modeling\\dash_parameter"
virtualenv --python=python3 venv
source venv/Scripts/activate
pip install -r requirements.txt

```


- heroku部署第一次运作
https://dash.plotly.com/deployment

```bash
#Initialize Heroku, add files to Git, and deploy
git init 
heroku create dashparameter # change my-dash-app to a unique name
git add .    # add all files to git
git commit -m 'Initial app boilerplate'
heroku git:remote -a dashparameter
heroku addons #查看addons
heroku addons:create heroku-postgresql:hobby-dev
set DATABASE_URL=postgres://$(whoami)
heroku pg:info
heroku pg:psql #本地和线上连接

PGUSER=postgres PGPASSWORD=password heroku pg:pull DATABASE_URL mylocaldb --app dashparameter
PGUSER=postgres PGPASSWORD=passwordheroku pg:push mylocaldb DATABASE_URL --app dashparameter

heroku config #获取databaseurl的详细信息
# heroku addons:docs heroku-postgresql ##查看文档
# heroku addons:open heroku-postgresql:hobby-dev
heroku pg:promote postgresql-cylindrical-87111 #提升为主数据库

git push heroku master # deploy code to heroku



heroku ps:scale web=1  # run the app with a 1 heroku "dyno"
heroku open
```

You should be able to view your app at https://dashparameter.herokuapp.com (changing my-dash-app to the name of your app).
https://dashparameter.herokuapp.com 

https://dashboard.heroku.com/ ##可查看database_url

# 非首次创建
```bash
cd "D:\\ck\\yzh\\2 replenishment\\8.modeling\\dash_parameter"
source venv/Scripts/activate
git init 
git status
git add .
git commit -m 'a description of the changes' #提交的时候写的话
heroku git:remote -a dashparameter #在本地仓库添加远程仓库A并将本地的master分支跟踪到远程的分支
git push heroku master # deploy code to heroku

heroku ps:scale web=1  # run the app with a 1 heroku "dyno"
heroku open

##internal error报错解决
# https://devcenter.heroku.com/articles/getting-started-with-python#start-a-console
可以通过heroku run命令打开控制台

$ heroku run python manage.py shell
$ heroku run python manage.py makemigrations        
$ heroku run python manage.py migrate
$ heroku run python manage.py createsuperuser
pipenv install gunicorn

python manage.py startapp dashparameter
python "D:\\ck\\yzh\\2 replenishment\\8.modeling\\dash_parameter\\manage.py" runserver
heroku run python manage.py shell ##初始化数据库
heroku run bash
ls
exit

```
heroku addons:create heroku-postgresql:hobby-dev
heroku pg:wait
heroku config #命令来确认应用的配置变量的名称和值。

##mysql迁移至pg
createdb pgdbname
pgloader mysql://username:password@localhost/mysqldbname postgresql:///pgdbname



5. Update the code and redeploy
git status # view the changes
git add .  # add all the changes
git commit -m 'a description of the changes'
git push heroku master
```

heroku login
heroku login -i   #登录Heroku账号，需要输入账号和密码
yzhmry1314@gmail.com
yzhmry@555

git init
heroku git:remote -a dashparameter
git add .
<!-- git commit -am "v1" -->
git commit -m 'Initial app boilerplate'
git push heroku master
heroku ps:scale web=1
heroku open



heroku keys:add    #添加SSH密匙
Heroku create    #在Heroku中创建新应用
git push heroku master    #使用git推送到Heroku主分支
python index.py

```



## Built With

- [Dash](https://dash.plot.ly/) - Main server and interactive components
- [Plotly Python](https://plot.ly/python/) - Used to create the interactive plots

