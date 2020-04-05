mutable struct model{
        RAM,
        PAR,
        DATA,
        REG,
        LOGL,
        OPT,
        EST,
        OBS_COV,
        IMP_COV,
        OBS_MEAN,
        OPT_RESULT,
        PAR_UNC}
    ram::RAM
    par::PAR
    data::DATA
    reg::REG
    logl::LOGL
    opt::OPT
    est::EST
    obs_cov::OBS_COV
    imp_cov::IMP_COV
    obs_mean::OBS_MEAN
    opt_result::OPT_RESULT
    par_unc::PAR_UNC
    model{RAM, PAR, DATA, REG, LOGL, OPT, EST, OBS_COV, IMP_COV, OBS_MEAN, OPT_RESULT, PAR_UNC}(
            ram, par, data,
            reg,
            logl,
            opt,
            est,
            obs_cov,
            imp_cov,
            obs_mean,
            opt_result,
            par_unc) where {
            RAM,
            PAR,
            DATA,
            REG,
            LOGL,
            OPT,
            EST,
            OBS_COV,
            IMP_COV,
            OBS_MEAN,
            OPT_RESULT,
            PAR_UNC} =
    new(ram, par, data,
            reg,
            logl,
            opt,
            est,
            obs_cov,
            imp_cov,
            obs_mean,
            opt_result,
            par_unc)
end


# second outer constructor for @set
model(ram::RAM, par::PAR, data::DATA = nothing,
        reg::REG = nothing,
        logl::LOGL = nothing,
        opt::OPT = "LBFGS",
        est::EST = ML,
        obs_cov::OBS_COV = nothing,
        imp_cov::IMP_COV = nothing,
        obs_mean::OBS_MEAN = nothing,
        opt_result::OPT_RESULT = nothing,
        par_unc::PAR_UNC = nothing) where {
        RAM,
        PAR,
        DATA,
        REG,
        LOGL,
        OPT,
        EST,
        OBS_COV,
        IMP_COV,
        OBS_MEAN,
        OPT_RESULT,
        PAR_UNC} =
        model{RAM, PAR, DATA, REG, LOGL, OPT, EST, OBS_COV, IMP_COV, OBS_MEAN, OPT_RESULT, PAR_UNC}(
                ram, par, data,
                reg,
                logl,
                opt,
                est,
                obs_cov,
                imp_cov,
                obs_mean,
                opt_result,
                par_unc)
### untyped struct for user

mutable struct sem_model
        ram
        par
        data
        reg
        logl
        opt
        est
        obs_cov
        imp_cov
        obs_mean
        opt_result
        par_unc
end

sem_model(ram, par;
        data = nothing,
        reg = nothing,
        logl = nothing,
        opt = "LBFGS",
        est = ML,
        obs_cov = nothing,
        imp_cov = nothing,
        obs_mean = nothing,
        opt_result = nothing,
        par_unc = nothing) =
        sem_model(ram, par,
                data,
                reg,
                logl,
                opt,
                est,
                obs_cov,
                imp_cov,
                obs_mean,
                opt_result,
                par_unc)



model(sem::sem_model) = model(sem.ram, sem.par,
                        sem.data,
                        sem.reg,
                        sem.logl,
                        sem.opt,
                        sem.est,
                        sem.obs_cov,
                        sem.imp_cov,
                        sem.obs_mean,
                        sem.opt_result,
                        sem.par_unc)


mutable struct reg{LASSO, LASSO_PEN, RIDGE, RIDGE_PEN}
        lasso::LASSO
        lasso_pen::LASSO_PEN
        ridge::RIDGE
        ridge_pen::RIDGE_PEN
        reg{LASSO, LASSO_PEN, RIDGE, RIDGE_PEN}(
                lasso, lasso_pen, ridge, ridge_pen) where {
                LASSO, LASSO_PEN, RIDGE, RIDGE_PEN} =
        new(lasso, lasso_pen, ridge, ridge_pen)
end


reg(;lasso::LASSO = nothing, lasso_pen::LASSO_PEN = nothing,
        ridge::RIDGE = nothing, ridge_pen::RIDGE_PEN = nothing) where {
        LASSO, LASSO_PEN, RIDGE, RIDGE_PEN} =
        reg{LASSO, LASSO_PEN, RIDGE, RIDGE_PEN}(
                lasso, lasso_pen, ridge, ridge_pen
                )

mutable struct par_unc{SE, P, Z}
    se::SE
    p::P
    z::Z
    par_unc{SE, P, Z}(se, p, z) where {SE, P, Z} = new(se, p, z)
end
