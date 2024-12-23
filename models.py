# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class AuthGroup(models.Model):
    name = models.CharField(unique=True, max_length=150)

    class Meta:
        managed = False
        db_table = 'auth_group'


class AuthGroupPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)
    permission = models.ForeignKey('AuthPermission', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_group_permissions'
        unique_together = (('group', 'permission'),)


class AuthPermission(models.Model):
    name = models.CharField(max_length=255)
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING)
    codename = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'auth_permission'
        unique_together = (('content_type', 'codename'),)


class AuthUser(models.Model):
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    is_superuser = models.IntegerField()
    username = models.CharField(unique=True, max_length=150)
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    email = models.CharField(max_length=254)
    is_staff = models.IntegerField()
    is_active = models.IntegerField()
    date_joined = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'auth_user'


class AuthUserGroups(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_groups'
        unique_together = (('user', 'group'),)


class AuthUserUserPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    permission = models.ForeignKey(AuthPermission, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_user_permissions'
        unique_together = (('user', 'permission'),)


class DjangoAdminLog(models.Model):
    action_time = models.DateTimeField()
    object_id = models.TextField(blank=True, null=True)
    object_repr = models.CharField(max_length=200)
    action_flag = models.PositiveSmallIntegerField()
    change_message = models.TextField()
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING, blank=True, null=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'django_admin_log'


class DjangoContentType(models.Model):
    app_label = models.CharField(max_length=100)
    model = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'django_content_type'
        unique_together = (('app_label', 'model'),)


class DjangoMigrations(models.Model):
    id = models.BigAutoField(primary_key=True)
    app = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    applied = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_migrations'


class DjangoSession(models.Model):
    session_key = models.CharField(primary_key=True, max_length=40)
    session_data = models.TextField()
    expire_date = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_session'


class User(models.Model):
    user_id = models.AutoField(primary_key=True)
    login_id = models.CharField(unique=True, max_length=50)
    username = models.CharField(max_length=20)
    userpwd = models.CharField(max_length=128)
    last_login = models.DateTimeField(null=True)
    is_superuser = models.BooleanField(default=False)
    is_staff = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    date_joined = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'user'


class Goods(models.Model):
    goodsKey = models.AutoField(primary_key=True)
    ratingCount = models.FloatField(null=True)
    goodsImg = models.CharField(max_length=500)
    ASIN = models.CharField(unique=True, max_length=30)
    goodsName = models.CharField(max_length=300)
    brand = models.CharField(max_length=50, null=True)
    originalPrice = models.IntegerField()
    discountedPrice = models.IntegerField()
    ratingAvg = models.FloatField(null=True)
    goodsInfo = models.TextField(null=True)
    goodsDesc = models.TextField(null=True)
    category1 = models.CharField(max_length=30)
    category2 = models.CharField(max_length=30)
    category3 = models.CharField(max_length=30)

    class Meta:
        managed = False
        db_table = 'goods'


class Orders(models.Model):
    orderKey = models.AutoField(primary_key=True)
    userKey = models.IntegerField()
    totalPrice = models.IntegerField()
    rdate = models.DateField()
    orderDetKey = models.IntegerField()
    goodsKey = models.ForeignKey(Goods, on_delete=models.CASCADE, db_column='goodsKey')
    price = models.IntegerField()
    cnt = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'orders'


class SituationCategory(models.Model):
    situationCateKey = models.AutoField(primary_key=True)
    situationCategory1 = models.CharField(max_length=30)
    situationCategory2 = models.CharField(max_length=30)
    situationCategory3 = models.CharField(max_length=30)

    class Meta:
        managed = False
        db_table = 'situationCategory'


class Situation(models.Model):
    situationKey = models.AutoField(primary_key=True)
    situationCateKey = models.ForeignKey(SituationCategory, on_delete=models.CASCADE, db_column='situationCateKey')
    headline1 = models.CharField(max_length=50)
    headline2 = models.CharField(max_length=50)
    mainKeyword = models.CharField(max_length=30)

    class Meta:
        managed = False
        db_table = 'situation'


class SituationKeyword(models.Model):
    situationKwKey = models.AutoField(primary_key=True)
    situationKey = models.ForeignKey(Situation, on_delete=models.CASCADE, db_column='situationKey')  # 상황 외래키
    situationKeyword = models.CharField(max_length=30)

    class Meta:
        managed = False
        db_table = 'situationKeyword'

class goodsKeyword(models.Model):
    goodsKwKey = models.AutoField(primary_key=True)
    ASIN = models.ForeignKey(Goods, on_delete=models.CASCADE, db_column='ASIN')
    goodsKeyword = models.CharField(max_length=30)

    class Meta:
        managed = False
        db_table = 'goodsKeyword'
