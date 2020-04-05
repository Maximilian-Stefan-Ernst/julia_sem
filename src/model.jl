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
        SE,
        Z,
        P}
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
    se::SE
    z::Z
    p::P
    model{RAM, PAR, DATA, REG, LOGL, OPT, EST, OBS_COV, IMP_COV, OBS_MEAN, OPT_RESULT, SE, Z, P}(
            ram, par, data,
            reg,
            logl,
            opt,
            est,
            obs_cov,
            imp_cov,
            obs_mean,
            opt_result,
            se,
            z,
            p) where {
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
            SE,
            Z,
            P} =
    new(ram, par, data,
            reg,
            logl,
            opt,
            est,
            obs_cov,
            imp_cov,
            obs_mean,
            opt_result,
            se,
            z,
            p)
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
        se::SE = nothing,
        z::Z = nothing,
        p::P = nothing) where {
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
        SE,
        Z,
        P} =
        model{RAM, PAR, DATA, REG, LOGL, OPT, EST, OBS_COV, IMP_COV, OBS_MEAN, OPT_RESULT, SE, Z, P}(
                ram, par, data,
                reg,
                logl,
                opt,
                est,
                obs_cov,
                imp_cov,
                obs_mean,
                opt_result,
                se,
                z,
                p)
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
        se
        z
        p
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
        se = nothing,
        z = nothing,
        p = nothing) =
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
                se,
                z,
                p)



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
                        sem.se,
                        sem.z,
                        sem.p)


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
